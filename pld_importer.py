bl_info = {
    "name": "RE3 PLD Importer",
    "author": "Your Name",
    "version": (1, 0),
    "blender": (4, 2, 0),
    "location": "File > Import > RE3 Model (.PLD)",
    "description": "Import Resident Evil 3 PLD model files",
    "warning": "",
    "doc_url": "",
    "category": "Import-Export",
}

import struct
from dataclasses import dataclass
import bpy
from bpy_extras.io_utils import ImportHelper
from bpy.props import StringProperty
from bpy.types import Operator
import json
from typing import List, Tuple, Optional, Dict, Any
import os

# Constantes
SCALE = 384  # Pour les UV
SCALE_VERT = 256  # Pour les vertices (Scale_)
MD2_MAGIC = 0x4098
OBJECT_MAGIC = 0x21C0

# Fonctions de logging
def log_debug(msg): print(f"[PLD Importer] DEBUG: {msg}")
def log_info(msg): print(f"[PLD Importer] INFO: {msg}")
def log_error(msg): print(f"[PLD Importer] ERROR: {msg}")
def log_warning(message: str):
    """Log un message d'avertissement"""
    print(f"[PLD Importer] WARNING: {message}")

@dataclass
class MD2Header:
    """Structure du header MD2 principal"""
    size: int          # DWORD (taille totale)
    num_parts: int     # DWORD (nombre de parts)

    def __init__(self, size: int, num_parts: int):
        self.size = size
        self.num_parts = num_parts

    @classmethod
    def from_bytes(cls, data: bytes) -> 'MD2Header':
        if len(data) < 8:
            log_error("MD2 header data too small")
            return None
            
        try:
            values = struct.unpack('<II', data[:8])
            header = cls(*values)
            
            log_info(f"MD2 Header: size=0x{header.size:04X}, parts={header.num_parts}")
            return header
            
        except struct.error as e:
            log_error(f"Failed to unpack MD2 header: {e}")
            return None

    def get_part_offset(self, part_index: int) -> int:
        """Calcule l'offset d'une part dans le fichier"""
        if part_index < 0 or part_index >= self.num_parts:
            return None
        # La taille de l'en-tête d'une part est de 24 octets
        return 8 + (part_index * 24)  # 8 pour l'en-tête MD2, 24 pour chaque part

@dataclass
class ObjectHeader:
    """Structure d'un header de part"""
    vertex_offset: int     # DWORD
    normal_offset: int     # DWORD
    vertex_count: int      # WORD
    triangle_offset: int   # DWORD
    quad_offset: int       # DWORD
    triangle_count: int    # WORD
    quad_count: int        # WORD

    @classmethod
    def from_bytes(cls, data: bytes):
        try:
            vertex_offset, normal_offset = struct.unpack('<II', data[:8])
            vertex_count = struct.unpack('<H', data[8:10])[0]
            # Skip 2 bytes (high word of vertex_count)
            triangle_offset, quad_offset = struct.unpack('<II', data[12:20])
            triangle_count, quad_count = struct.unpack('<HH', data[20:24])
            
            obj = cls(
                vertex_offset=vertex_offset,
                normal_offset=normal_offset,
                vertex_count=vertex_count,
                triangle_offset=triangle_offset,
                quad_offset=quad_offset,
                triangle_count=triangle_count,
                quad_count=quad_count
            )
            
            log_info(f"Part Header:")
            log_info(f"  Vertices: count={obj.vertex_count}, offset=0x{obj.vertex_offset:04X}")
            log_info(f"  Triangles: count={obj.triangle_count}, offset=0x{obj.triangle_offset:04X}")
            log_info(f"  Quads: count={obj.quad_count}, offset=0x{obj.quad_offset:04X}")
            
            return obj
            
        except struct.error as e:
            log_error(f"Failed to unpack object header: {e}")
            return None

    def is_bone_marker(self) -> bool:
        """Détermine si cette part est un marqueur de bone"""
        return (self.vertex_count == 3 and 
                self.triangle_count <= 1 and 
                self.quad_count == 0)

class MD2Mesh:
    """Représente un mesh MD2"""
    def __init__(self, vertices: List[Tuple[float, float, float]], triangles: List[Tuple[int, int, int]], uvs: List[Tuple[float, float]], normals: List[Tuple[float, float, float]]):
        self.vertices = vertices
        self.triangles = triangles
        self.uvs = uvs
        self.normals = normals

    @staticmethod
    def from_part(data: bytes, part_header: ObjectHeader) -> Optional['MD2Mesh']:
        """Crée un MD2Mesh à partir des données d'une part"""
        try:
            # Lecture des vertices
            vertices = read_vertices(data, part_header)
            if not vertices:
                return None
            
            # Lecture des normales
            normals = read_normals(data, part_header) 
            if not normals:
                return None

            triangles = []
            uvs = []
            
            # Lecture des triangles si présents
            if part_header.triangle_count > 0:
                tri_data = read_triangles(data, part_header.triangle_offset + 8, part_header.triangle_count)
                if isinstance(tri_data, tuple):
                    triangles.extend(tri_data[0])
                    uvs.extend(tri_data[1])

            # Lecture des quads si présents
            if part_header.quad_count > 0:
                quad_data = read_quads(data, part_header.quad_offset + 8, part_header.quad_count)
                if isinstance(quad_data, tuple):
                    triangles.extend(quad_data[0])
                    uvs.extend(quad_data[1])

            return MD2Mesh(vertices, triangles, uvs, normals)  # On passe les normales au constructeur

        except Exception as e:
            log_error(f"Failed to create MD2Mesh: {str(e)}")
            return None

def load_tim_texture(filepath: str) -> Optional[bpy.types.Image]:
    """Charge une texture TIM et la convertit en image Blender"""
    try:
        with open(filepath, 'rb') as f:
            data = f.read()
            
        # Vérification du magic number TIM (0x10)
        if data[0] != 0x10:
            log_error("Not a valid TIM file")
            return None
            
        # Lecture du header TIM
        bpp_flag = data[4]  # Bits per pixel flag
        has_clut = (bpp_flag & 0x8) != 0
        bpp = 4 if (bpp_flag & 0x3) == 0 else 8  # 4 ou 8 bits par pixel
        
        offset = 8
        
        # Si on a une CLUT, on la lit
        if has_clut:
            clut_size = struct.unpack('<H', data[offset+4:offset+6])[0]
            clut_width = struct.unpack('<H', data[offset+8:offset+10])[0]
            clut_height = struct.unpack('<H', data[offset+10:offset+12])[0]
            offset += 12 + (clut_width * clut_height * 2)  # Skip CLUT data
            
        # Lecture des dimensions de l'image
        img_size = struct.unpack('<I', data[offset:offset+4])[0]
        img_width = struct.unpack('<H', data[offset+8:offset+10])[0] * 2  # *2 car stocké en 16-bit words
        img_height = struct.unpack('<H', data[offset+10:offset+12])[0]
        
        # Création de l'image Blender
        image_name = os.path.basename(filepath)
        if image_name in bpy.data.images:
            bpy.data.images.remove(bpy.data.images[image_name])
        
        image = bpy.data.images.new(image_name, width=img_width, height=img_height, alpha=True)
        
        # TODO: Conversion des pixels TIM en pixels RGBA
        # Pour l'instant, on met juste une texture grise
        pixels = [0.5, 0.5, 0.5, 1.0] * (img_width * img_height)
        image.pixels = pixels
        
        return image
        
    except Exception as e:
        log_error(f"Failed to load TIM texture: {str(e)}")
        return None

def create_material(texture_image: bpy.types.Image) -> bpy.types.Material:
    """Crée un matériau avec la texture donnée"""
    mat = bpy.data.materials.new(name="MD2_Material")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    
    # Nettoyage des nodes existants
    nodes.clear()
    
    # Création des nodes
    node_tex = nodes.new('ShaderNodeTexImage')
    node_tex.image = texture_image
    
    node_princ = nodes.new('ShaderNodeBsdfPrincipled')
    node_princ.inputs['Specular IOR Level'].default_value = 1.0
    node_princ.inputs['Roughness'].default_value = 1.0
    
    node_output = nodes.new('ShaderNodeOutputMaterial')
    
    # Connexions
    links = mat.node_tree.links
    links.new(node_tex.outputs['Color'], node_princ.inputs['Base Color'])
    links.new(node_tex.outputs['Alpha'], node_princ.inputs['Alpha'])
    links.new(node_princ.outputs['BSDF'], node_output.inputs['Surface'])
    
    return mat

def process_md2_file(data: bytes, tim_filepath: str = None) -> List[bpy.types.Object]:
    """Traite un fichier MD2 avec sa texture optionnelle"""
    objects = []
    # Chargement de la texture si spécifiée
    texture_image = None
    material = None
    if tim_filepath and os.path.exists(tim_filepath):
        texture_image = load_tim_texture(tim_filepath)
        if texture_image:
            material = create_material(texture_image)
    
    try:
        header = MD2Header.from_bytes(data)
        if not header:
            return []
            
        log_info(f"MD2 Header: size={header.size:X}, parts={header.num_parts}")
        bone_data = []
        
        for i in range(header.num_parts):
        #for i in range(1):
            offset = header.get_part_offset(i)
            part_header = ObjectHeader.from_bytes(data[offset:])
            if not part_header:
                continue
                
            log_info(f"Part Header:")
            log_info(f"  Vertices: count={part_header.vertex_count}, offset={part_header.vertex_offset:X}")
            log_info(f"  Triangles: count={part_header.triangle_count}, offset={part_header.triangle_offset:X}")
            log_info(f"  Quads: count={part_header.quad_count}, offset={part_header.quad_offset:X}")
            
            # Détermine si c'est un marqueur de bone
            is_marker = part_header.is_bone_marker()
            if is_marker:
                log_info(f"Part {i} identified as bone marker")
                
                # Pour les bone markers, on crée quand même un mesh mais on stocke les infos
                mesh = MD2Mesh.from_part(data, part_header)
                if not mesh:
                    continue
                    
                bone_info = {
                    'part_index': i,
                    'offset': offset,
                    'header': {
                        'vertex_count': part_header.vertex_count,
                        'vertex_offset': part_header.vertex_offset,
                        'triangle_count': part_header.triangle_count,
                        'triangle_offset': part_header.triangle_offset,
                        'quad_count': part_header.quad_count,
                        'quad_offset': part_header.quad_offset
                    },
                    'vertices': mesh.vertices,
                    'triangles': mesh.triangles,
                    'raw_data': data[offset:offset + part_header.vertex_offset + part_header.vertex_count * 12]
                }
                bone_data.append(bone_info)
            else:
                # Pour les meshes normaux
                mesh = MD2Mesh.from_part(data, part_header)
                if not mesh:
                    continue
            
            # Création du mesh Blender
            part = {
                'vertices': mesh.vertices,
                'triangles': mesh.triangles,
                'uvs': mesh.uvs,
                'normals': mesh.normals,
                'is_bone_marker': is_marker
            }
            
            # Debug des données avant création du mesh
            log_info(f"Part {i} data before mesh creation:")
            log_info(f"  Vertices: {len(mesh.vertices)} items, first={mesh.vertices[0] if mesh.vertices else 'None'}")
            log_info(f"  Triangles: {len(mesh.triangles)} items, first={mesh.triangles[0] if mesh.triangles else 'None'}")
            log_info(f"  UVs: {len(mesh.uvs)} items, first={mesh.uvs[0] if mesh.uvs else 'None'}")
            
            obj = create_mesh_from_part(part, i)
            if obj:
                objects.append(obj)
                # Ajout du matériau à l'objet
                if material:
                    if not obj.data.materials:  # Si l'objet n'a pas déjà un matériau
                        obj.data.materials.append(material)
                if is_marker:
                    obj['bone_data'] = bone_info
                
            log_info(f"Part {i}: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles {'(bone marker)' if is_marker else ''}")
        
        return objects
        
    except Exception as e:
        log_error(f"Failed to process MD2 file: {str(e)}")
        return []

@dataclass
class Vertex:
    """Structure d'un vertex (aligné sur DWORD)"""
    x: int  # short (2 bytes)
    y: int  # short (2 bytes)
    z: int  # short (2 bytes)
    pad: int  # short (2 bytes) pour alignement DWORD

    @classmethod
    def from_bytes(cls, data: bytes):
        try:
            x, y, z, pad = struct.unpack('<hhhh', data)
            return cls(x, y, z, pad)
        except struct.error:
            return None
        
def read_normals(data: bytes, header: ObjectHeader) -> List[Tuple[float, float, float]]:
    """Lit les normales depuis les données binaires"""
    SCALE_VERT = 256.0
    
    normals = []
    offset = header.normal_offset + 8  # On commence au deuxième vertex
    count = header.vertex_count 

    log_info(f"Reading {count} normal from offset {offset}")
    
    try:
        valid_count = 0
        for i in range(count):
            pos = offset + (i * 8)
            if pos + 8 > len(data):
                break
            
            x, y, z, pad = struct.unpack('<hhhh', data[pos:pos + 8])
            
            # Conversion avec rotation de 90° autour de X : (-z, -y, -x) -> (-z, x, y)
            vertex = (-z / SCALE_VERT, x / SCALE_VERT, -y / SCALE_VERT)
            normals.append(vertex)
            valid_count += 1
        
        log_info(f"Successfully read {valid_count} normals")
        return normals
    
    except struct.error as e:
        log_error(f"Failed to read normals: {str(e)}")
        return []

def read_vertices(data: bytes, header: ObjectHeader) -> List[Tuple[float, float, float]]:
    """Lit les vertices selon la structure TVertex"""
    SCALE_VERT = 256.0
    
    vertices = []
    offset = header.vertex_offset + 8  # On commence au deuxième vertex
    count = header.vertex_count 
    
    log_info(f"Reading {count} vertices from offset {offset}")
    
    try:
        valid_count = 0
        for i in range(count):
            pos = offset + (i * 8)
            if pos + 8 > len(data):
                break
            
            x, y, z, pad = struct.unpack('<hhhh', data[pos:pos + 8])
            
            # Conversion avec rotation de 90° autour de X : (-z, -y, -x) -> (-z, x, y)
            vertex = (-z / SCALE_VERT, x / SCALE_VERT, -y / SCALE_VERT)
            vertices.append(vertex)
            valid_count += 1
        
        log_info(f"Successfully read {valid_count} vertices")
        return vertices
        
    except struct.error as e:
        log_error(f"Failed to read vertices: {str(e)}")
        return []

def read_triangles(data: bytes, offset: int, count: int) -> Tuple[List[Tuple[int, int, int]], List[Tuple[float, float]]]:
    """Lit les triangles selon la structure TTriangle"""
    SCALE_U = 384.0
    SCALE_V = 256.0
    
    triangles = []
    uvs = []
    
    log_info(f"Reading {count} triangles")
    
    for i in range(count):
        tri_offset = offset + (i * 12)
        
        # Lecture selon la structure TTriangle
        u0 = data[tri_offset + 0]
        v0 = data[tri_offset + 1]
        clut = struct.unpack('<H', data[tri_offset + 2:tri_offset + 4])[0]
        u1 = data[tri_offset + 4]
        v1 = data[tri_offset + 5]
        page = data[tri_offset + 6]
        vi0 = data[tri_offset + 7]
        u2 = data[tri_offset + 8]
        v2 = data[tri_offset + 9]
        vi1 = data[tri_offset + 10]
        vi2 = data[tri_offset + 11]
        
        # Lecture du tpage et calcul de l'offset
        tpage = int.from_bytes(data[tri_offset + 6:tri_offset + 8], 'little')
        page_offset = (tpage & 0x1F) * (128.0 / 384.0)
        
        # Conversion des UV avec le tpage
        uv0 = (u0 / 384.0 + page_offset, -v0 / SCALE_V)
        uv1 = (u1 / 384.0 + page_offset, -v1 / SCALE_V)
        uv2 = (u2 / 384.0 + page_offset, -v2 / SCALE_V)
        
        triangles.append((vi0, vi1, vi2))
        uvs.extend([uv0, uv1, uv2])
    
    log_info(f"Successfully read {len(triangles)} triangles")
    return triangles, uvs

def read_quads(data: bytes, offset: int, count: int) -> Tuple[List[Tuple[int, int, int]], List[Tuple[float, float]]]:
    """Lit les quads selon la structure TQuad"""
    SCALE_U = 384.0
    SCALE_V = 256.0
    
    triangles = []
    uvs = []
    
    log_info(f"Reading {count} quads")
    
    for i in range(count):
        quad_offset = offset + (i * 16)
        
        # Lecture selon la structure TQuad
        u0 = data[quad_offset + 0]
        v0 = data[quad_offset + 1]
        clut = struct.unpack('<H', data[quad_offset + 2:quad_offset + 4])[0]
        u1 = data[quad_offset + 4]
        v1 = data[quad_offset + 5]
        page = struct.unpack('<H', data[quad_offset + 6:quad_offset + 8])[0]
        u2 = data[quad_offset + 8]
        v2 = data[quad_offset + 9]
        vi0 = data[quad_offset + 10]
        vi1 = data[quad_offset + 11]
        u3 = data[quad_offset + 12]
        v3 = data[quad_offset + 13]
        vi2 = data[quad_offset + 14]
        vi3 = data[quad_offset + 15]
        
        # Lecture du tpage et calcul de l'offset
        tpage = int.from_bytes(data[quad_offset + 6:quad_offset + 8], 'little')
        page_offset = (tpage & 0x1F) * (128.0 / 384.0)
        
        # Conversion des UV avec le tpage
        uv0 = (u0 / 384.0 + page_offset, -v0 / SCALE_V)
        uv1 = (u1 / 384.0 + page_offset, -v1 / SCALE_V)
        uv2 = (u2 / 384.0 + page_offset, -v2 / SCALE_V)
        uv3 = (u3 / 384.0 + page_offset, -v3 / SCALE_V)
        
        # Triangulation du quad
        triangles.append((vi0, vi1, vi2))
        triangles.append((vi1, vi3, vi2))
        uvs.extend([uv0, uv1, uv2])
        uvs.extend([uv1, uv3, uv2])
    
    log_info(f"Successfully converted {count} quads to {len(triangles)} triangles")
    return triangles, uvs

def create_mesh(vertices, triangles):
    """Crée un mesh Blender depuis les vertices et triangles"""
    try:
        # Debug des données d'entrée
        log_info(f"Creating mesh from {len(vertices)} vertices and {len(triangles)} triangles")
        if len(vertices) > 0:
            log_debug(f"First vertex: {vertices[0]}")
            log_debug(f"Last vertex: {vertices[-1]}")
        if len(triangles) > 0:
            log_debug(f"First triangle: {triangles[0]}")
            log_debug(f"Last triangle: {triangles[-1]}")
        
        # Création du mesh
        mesh = bpy.data.meshes.new("RE3_Mesh")
        obj = bpy.data.objects.new("RE3_Object", mesh)
        
        # Conversion des vertices avec le bon scale
        verts = []
        for x, y, z in vertices:
            # Utilisation de SCALE_VERT (256) au lieu de SCALE (384)
            scaled_x = float(x) / SCALE_VERT
            scaled_y = float(-y) / SCALE_VERT  # Inversion Y
            scaled_z = float(-z) / SCALE_VERT  # Inversion Z
            
            if len(verts) < 5:
                log_debug(f"Vertex conversion: ({x},{y},{z}) -> ({scaled_x:.4f},{scaled_y:.4f},{scaled_z:.4f})")
            verts.append((scaled_x, scaled_y, scaled_z))
        
        # Création des faces
        faces = []
        for v1, v2, v3 in triangles:
            if v1 < len(verts) and v2 < len(verts) and v3 < len(verts):
                faces.append((int(v1), int(v2), int(v3)))
            else:
                log_error(f"Invalid triangle: ({v1},{v2},{v3}), max vertex index: {len(verts)-1}")
        
        # Debug final
        log_info(f"Final mesh data: {len(verts)} vertices, {len(faces)} faces")
        
        # Création du mesh
        mesh.from_pydata(verts, [], faces)
        mesh.validate()
        mesh.update()
        
        # Ajout à la scène
        bpy.context.collection.objects.link(obj)
        
        # Sélection et focus
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        
        return obj
        
    except Exception as e:
        log_error(f"Failed to create mesh: {e}")
        import traceback
        log_error(traceback.format_exc())
        return None

class TIMTexture:
    def __init__(self, data: bytes):
        self.data = data
        self.width = 0
        self.height = 0
        self.has_clut = False
        self.bpp = 0
        self.clut_data = []  # Liste des 3 palettes
        self.image_data = None
        
    def read_header(self) -> bool:
        try:
            # Debug du header
            log_info(f"TIM data size: {len(self.data)} bytes")
            log_info(f"First 16 bytes: {' '.join([f'{x:02X}' for x in self.data[:16]])}")
            
            if self.data[0] != 0x10:
                log_error(f"Invalid TIM magic number: {self.data[0]:02X}")
                return False
                
            format_flag = self.data[4]
            self.has_clut = (format_flag & 0x08) != 0
            self.bpp = 4 if (format_flag & 0x03) == 0 else 8 if (format_flag & 0x03) == 1 else 16
            
            log_info(f"Format: {self.bpp}bpp, has CLUT: {self.has_clut}")
            
            offset = 8
            
            # Lecture des palettes
            if self.has_clut:
                clut_size = int.from_bytes(self.data[offset:offset+4], 'little')
                clut_x = int.from_bytes(self.data[offset+4:offset+6], 'little')
                clut_y = int.from_bytes(self.data[offset+6:offset+8], 'little')
                clut_w = int.from_bytes(self.data[offset+8:offset+10], 'little')
                clut_h = int.from_bytes(self.data[offset+10:offset+12], 'little')
                
                log_info(f"CLUT header: size={clut_size}, pos=({clut_x},{clut_y}), dim={clut_w}x{clut_h}")
                
                offset += 12
                clut_data_size = clut_size - 12
                self.clut_data = []
                
                # Lecture des données CLUT brutes
                raw_clut = self.data[offset:offset + clut_data_size]
                log_info(f"Raw CLUT size: {len(raw_clut)} bytes")
                
                # Découpage en palettes selon la hauteur
                palette_size = clut_w * 2  # 2 bytes par couleur
                for i in range(clut_h):
                    start = i * palette_size
                    end = start + palette_size
                    if end <= len(raw_clut):
                        palette = raw_clut[start:end]
                        self.clut_data.append(palette)
                        log_info(f"Palette {i}: size={len(palette)} bytes, first colors: {' '.join([f'{x:02X}' for x in palette[:8]])}")
            
                offset += clut_data_size
            
            # Lecture des données image
            image_size = int.from_bytes(self.data[offset:offset+4], 'little')
            image_x = int.from_bytes(self.data[offset+4:offset+6], 'little')
            image_y = int.from_bytes(self.data[offset+6:offset+8], 'little')
            raw_width = int.from_bytes(self.data[offset+8:offset+10], 'little')
            self.height = int.from_bytes(self.data[offset+10:offset+12], 'little')
            
            # Ajustement de la largeur selon le bpp
            self.width = raw_width * (16 // self.bpp) if self.bpp in [4, 8] else raw_width
            
            log_info(f"Image header: size={image_size}, pos=({image_x},{image_y}), raw_dim={raw_width}x{self.height}, final_dim={self.width}x{self.height}")
            
            offset += 12
            image_data_size = image_size - 12
            self.image_data = self.data[offset:offset + image_data_size]
            
            log_info(f"Image data: expected={image_data_size} bytes, got={len(self.image_data)} bytes")
            log_info(f"First pixels: {' '.join([f'{x:02X}' for x in self.image_data[:16]])}")
            
            # Dump complet des palettes
            for palette_index, palette in enumerate(self.clut_data):
                log_info(f"\nPalette {palette_index} complete dump:")
                for i in range(0, len(palette), 2):
                    if i + 1 < len(palette):
                        color = (palette[i+1] << 8) | palette[i]
                        a = "1" if (color & 0x8000) else "0"
                        r = ((color >> 0) & 0x1F)
                        g = ((color >> 5) & 0x1F)
                        b = ((color >> 10) & 0x1F)
                        log_info(f"  [{i//2:02X}] {a}:{b:02X}{g:02X}{r:02X} (raw=0x{color:04X})")
            
            return True
            
        except Exception as e:
            log_error(f"Failed to read TIM header: {str(e)}")
            return False

    def create_blender_texture(self) -> Optional[bpy.types.Image]:
        try:
            image_name = "TIM_Texture"
            if image_name in bpy.data.images:
                bpy.data.images.remove(bpy.data.images[image_name])
                
            image = bpy.data.images.new(image_name, width=self.width, height=self.height, alpha=True)
            pixels = []
            
            if self.bpp == 8 and self.has_clut:
                # Log de la palette
                log_info("CLUT (first 16 entries of first palette):")
                for i in range(min(16, len(self.clut_data[0]) // 2)):
                    color = (self.clut_data[0][i*2+1] << 8) | self.clut_data[0][i*2]
                    r = ((color >> 0) & 0x1F) / 31.0
                    g = ((color >> 5) & 0x1F) / 31.0
                    b = ((color >> 10) & 0x1F) / 31.0
                    a = 0.0 if (color & 0x8000) else 1.0
                    log_info(f"  [{i:02X}] color=0x{color:04X}, r={r:.2f}, g={g:.2f}, b={b:.2f}, a={a:.2f}")
                
                # 8bpp avec CLUT (256 couleurs)
                iy = self.height - 1
                for y in range(self.height):
                    for x in range(self.width):
                        io = (iy * self.width) + x
                        if io < len(self.image_data):
                            idx = self.image_data[io]
                            # Sélectionner la palette en fonction de x
                            palette_index = x // 128
                            if palette_index > 2:
                                palette_index = 2
                                
                            clut_offset = idx * 2
                            if clut_offset + 1 < len(self.clut_data[palette_index]):
                                color = (self.clut_data[palette_index][clut_offset+1] << 8) | self.clut_data[palette_index][clut_offset]
                                r = ((color >> 0) & 0x1F) / 31.0
                                g = ((color >> 5) & 0x1F) / 31.0
                                b = ((color >> 10) & 0x1F) / 31.0
                                a = 0.0 if (color & 0x8000) else 1.0
                                
                                if x < 8 and y == 0:
                                    log_info(f"  -> x={x} palette={palette_index} color=0x{color:04X}, r={r:.2f}, g={g:.2f}, b={b:.2f}, a={a:.2f}")
                            else:
                                r, g, b, a = 0, 0, 0, 1
                            pixels.extend([r, g, b, a])
                    iy -= 1
            
            # Remplir le reste si nécessaire
            while len(pixels) < self.width * self.height * 4:
                pixels.extend([0, 0, 0, 1])
            
            log_info(f"Created texture: {self.width}x{self.height} pixels ({len(pixels)} values)")
            image.pixels = pixels[:self.width * self.height * 4]
            image.pack()
            
            return image
            
        except Exception as e:
            log_error(f"Failed to create texture: {str(e)}")
            return None

class Skeleton:
    """Classe pour gérer le squelette"""
    def __init__(self, data: bytes):
        self.data = data
        self.bones = []

    def read_skeleton(self):
        try:
            # TODO: Implémenter la lecture du squelette
            # Format à déterminer d'après le code Pascal
            pass
            
        except struct.error as e:
            log_error(f"Failed to read skeleton: {e}")
            return False

def process_pld_file(data: bytes):
    """Traite un fichier PLD complet"""
    if len(data) < 8:
        log_error("File too small")
        return None
        
    # Lecture du header (8 premiers bytes)
    size, file_count = struct.unpack('<II', data[:8])
    log_info(f"PLD: size={size}, files={file_count}")
    
    # Lecture des offsets à la fin du fichier
    offset_pos = len(data) - (file_count * 4)
    offsets = list(struct.unpack(f'<{file_count}I', data[offset_pos:offset_pos + file_count * 4]))
    log_info(f"Offsets from end of file: {[f'0x{x:08X}' for x in offsets]}")
    
    # Extraction des fichiers
    files = []
    for i in range(file_count):
        start = offsets[i]
        end = offsets[i+1] if i < file_count-1 else offset_pos
        file_data = data[start:end]
        files.append(file_data)
        log_info(f"File {i}: offset=0x{start:08X}, size={len(file_data)}")
        if len(file_data) > 0:
            log_info(f"First bytes: {' '.join([f'{b:02X}' for b in file_data[:16]])}")
    
    # Traitement des fichiers dans l'ordre:
    # 0: EDD (squelette)
    # 1: EMR
    # 2: MD2 (mesh)
    # 3: DAT
    # 4: TIM (texture)
    
    mesh_obj = None
    material = None
    
    if len(files) >= 5:
        # Traitement du TIM d'abord pour créer le matériau
        if len(files[4]) > 0:  # TIM
            log_info("Processing TIM texture file...")
            texture = TIMTexture(files[4])
            if texture.read_header():
                texture_image = texture.create_blender_texture()
                if texture_image:
                    material = create_material(texture_image)
        
        # Traitement du MD2 ensuite
        if len(files[2]) > 0:  # MD2
            log_info("Processing MD2 mesh file...")
            objects = process_md2_file(files[2])
            if material and objects:
                log_info("Applying material to objects...")
                for obj in objects:
                    if obj.type == 'MESH':
                        if not obj.data.materials:
                            obj.data.materials.append(material)
            mesh_obj = objects
        
        return mesh_obj
    
    return None

def import_pld(filepath: str):
    """Point d'entrée principal pour l'import PLD"""
    try:
        log_info(f"Starting PLD import from: {filepath}")
        
        # 1. Lecture du fichier PLD
        with open(filepath, 'rb') as f:
            pld_data = f.read()
            
        # 2. Extraction et traitement du PLD
        mesh_obj = process_pld_file(pld_data)
        if not mesh_obj:
            log_error("Failed to process PLD file")
            return False
            
        log_info("PLD import completed successfully")
        return True
        
    except Exception as e:
        log_error(f"Failed to import PLD: {str(e)}")
        return False

def register():
    bpy.utils.register_class(ImportPLD)
    bpy.types.TOPBAR_MT_file_import.append(menu_func_import)

def unregister():
    bpy.utils.unregister_class(ImportPLD)
    bpy.types.TOPBAR_MT_file_import.remove(menu_func_import)

def menu_func_import(self, context):
    self.layout.operator(ImportPLD.bl_idname, text="RE3 Model (.PLD)")

def is_bone_marker(part_header):
    """Détermine si une part est un marqueur de bone"""
    return (part_header['vertex_count'] == 3 and 
            part_header['triangle_count'] <= 1 and 
            part_header['quad_count'] == 0)

def create_mesh_from_part(part: dict, part_index: int) -> Optional[bpy.types.Object]:
    """Crée un mesh Blender à partir d'une part"""
    try:
        # Création du mesh
        mesh = bpy.data.meshes.new(f"MD2_Mesh_{part_index}")
        obj = bpy.data.objects.new(f"MD2_Object_{part_index}", mesh)
        
        # Ajout à la scène
        bpy.context.collection.objects.link(obj)
        
        # Création des vertices et faces
        vertices = part['vertices']
        triangles = part['triangles']
        uvs = part['uvs']
        normals = part.get('normals', [])  # Récupération des normales
        
        # Assignation des données au mesh
        mesh.from_pydata(vertices, [], triangles)
        mesh.update()

        # Application des normales si présentes
        if normals and len(normals) == len(vertices):
            # Créer une liste de normales pour chaque loop
            loop_normals = []
            for face in mesh.polygons:
                for vertex_idx in face.vertices:
                    loop_normals.append(normals[vertex_idx])
            
            mesh.normals_split_custom_set(loop_normals)
        
        # Création des UV
        if len(uvs) > 0:
            log_info(f"Creating UV map with {len(uvs)} coordinates")
            uv_layer = mesh.uv_layers.new(name="UVMap")
            for face in mesh.polygons:
                for i, loop_idx in enumerate(face.loop_indices):
                    # Utilise l'index de la face * nombre de vertices par face + position dans la face
                    uv_idx = face.index * len(face.vertices) + i
                    if uv_idx < len(uvs):
                        uv = uvs[uv_idx]
                        log_debug(f"Face {face.index} vertex {i} UV: {uv[0]:.3f}, {uv[1]:.3f}")
                        uv_layer.data[loop_idx].uv = uv
        
        return obj
        
    except Exception as e:
        log_error(f"Failed to create mesh: {str(e)}")
        return None

def get_object_override(active_object, objects: list = None):
    """Helper pour créer un override de contexte"""
    if objects is None:
        objects = [active_object]
    elif active_object not in objects:
        objects.append(active_object)

    return {
        "selected_objects": objects,
        "selected_editable_objects": objects,
        "active_object": active_object,
        "object": active_object,
    }

class ImportPLD(bpy.types.Operator, ImportHelper):
    """Import from PLD file format"""
    bl_idname = "import_mesh.pld"
    bl_label = "Import PLD"
    filename_ext = ".PLD"
    
    filter_glob: bpy.props.StringProperty(
        default="*.PLD",
        options={'HIDDEN'},
    )
    
    def execute(self, context):
        if import_pld(self.filepath):
            return {'FINISHED'}
        return {'CANCELLED'}

if __name__ == "__main__":
    register()
