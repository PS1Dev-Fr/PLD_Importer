# RE3 PS1 PLD Blender Plugin 

Un plugin Blender pour importer les fichiers **PLD** du jeu _Resident Evil 3_ (version PlayStation 1). Ce plugin permet d'importer les personnages du jeu directement dans Blender.

## Fonctionnalités

- **Importation des fichiers PLD** : Les fichiers de personnages (_characters_) de RE3 sont convertis et affichés dans Blender.
- **Support des meshes et textures** : Les modèles et leurs textures associées sont importés avec précision.
- **Limitation actuelle** : Le squelette (_rigging_) n'est pas encore pris en charge, mais son intérêt reste à évaluer.

## Installation

1. Téléchargez le fichier `.py` du plugin depuis ce dépôt.
2. Ouvrez Blender, puis rendez-vous dans les **Préférences** :
   - **Édition > Préférences > Add-ons > Installer**.
3. Sélectionnez le fichier du plugin et activez-le en cochant la case correspondante.
4. Le plugin est maintenant disponible dans l'interface Blender.

## Utilisation

1. Lancez Blender et ouvrez un nouveau projet.
2. Dans le menu supérieur, sélectionnez **Fichier > Importer > RE3 PLD (.pld)**.
3. Choisissez un fichier PLD depuis votre répertoire de ressources du jeu _Resident Evil 3 PS1_.
4. Les meshes et textures du personnage seront importés dans la scène active.

## Roadmap

- [ ] Ajouter le support du rigging (importation des squelettes).
- [ ] Améliorer la gestion des textures pour prendre en charge les variations d’éclairage.
- [ ] Implémenter un exporteur pour faciliter les modifications des modèles.
- [ ] Documentation détaillée pour les développeurs souhaitant contribuer.

## Contributions

Les contributions sont les bienvenues ! Si vous souhaitez améliorer ce plugin, n'hésitez pas à ouvrir une _issue_ ou soumettre une _pull request_.

## Avertissement

Ce plugin est destiné à des fins éducatives et personnelles uniquement. L'utilisation des ressources du jeu _Resident Evil 3 PS1_ est soumise aux droits d'auteur de Capcom. Veuillez respecter les lois en vigueur sur la propriété intellectuelle.
