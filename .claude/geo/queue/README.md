# Files d'exécution

Une file = un lot homogène : même skill, même produit, même langue. Format d'un fichier `<lot>.md` :

```
# Lot : <nom> : produit : <nom> : langue : <fr|en>
# Skill : <comparison-page | statistics-article | foundational-article | free-tool | internal-linking | directory-submission>
# Règle : un item passe de [ ] à [x] quand il est terminé, avec le chemin du livrable.

- [ ] item-001 | <titre ou slug> | requête : <…> | cluster : <…> | livrable :
- [ ] item-002 | …
```

Un fichier `notes.md` à côté garde les décisions et conventions découvertes pendant le lot, pour que chaque itération en profite.

Exécution : voir le skill `ralph-loop`. Moins d'une trentaine d'items : sous-agents en parallèle. Au-delà, ou pendant la nuit : boucle externe, une session par item.

Garde-fou : chaque item passe la Definition of Done du playbook. Les zones sensibles restent marquées `[FACT-CHECK HUMAIN]` et rien ne se publie automatiquement.
