# Guides des outils : index et état des démonstrations

Les 12 articles existent en français et en anglais, avec 67 captures distinctes référencées. Ils sont configurés pour publication (`draft: false`), avec liens réciproques depuis la page des outils, exemples reproductibles et recommandations de lecture. Les versions de chaque langue sont reliées automatiquement par hreflang.

## Reprise éditoriale du 28 octobre

Le premier outil, Summaries, a été réécrit autour de Sagittarius A* avec un parcours propre au papier, huit captures par langue, une preuve issue du corps de l’article et la galerie des six figures. Le deuxième outil, Discuter avec les articles, a été repris avec les papiers I et VI sur Sagittarius A*, une analyse visuelle réellement exécutée, des preuves consultables et un tableau à trois colonnes. Il comporte neuf captures par langue. Les dix autres articles restent dans leur version précédente et devront être repris individuellement.

## Relecture

| Outil | Français | English | Démonstration réelle |
|---|---|---|---|
| Summaries | [Lire](../guides/fr/how-to-summarize-a-research-paper.md) | [Read](../guides/en/how-to-summarize-a-research-paper.md) | Repris le 28 octobre : Sagittarius A*, 2311.08680v2, CC-BY 4.0 ; lecture intégrale confirmée, six figures accessibles, huit captures distinctes par langue |
| Chat with papers | [Lire](../guides/fr/how-to-chat-with-research-papers.md) | [Read](../guides/en/how-to-chat-with-research-papers.md) | Repris le 28 octobre : deux papiers CC-BY lus intégralement, figure 3 transmise, relance, comparaison et tableau à six cellules sourcées, neuf captures par langue |
| Collections | [Lire](../guides/fr/organize-research-paper-collections.md) | [Read](../guides/en/organize-research-paper-collections.md) | Collection dédiée et tableaux sourcés terminés dans les deux langues |
| Daily digest | [Lire](../guides/fr/daily-digest-research-papers.md) | [Read](../guides/en/daily-digest-research-papers.md) | Édition web française du 27 octobre ; aperçu non enregistré réellement configuré en anglais |
| Saved searches | [Lire](../guides/fr/saved-searches-research-alerts.md) | [Read](../guides/en/saved-searches-research-alerts.md) | Requête sauvegardée et aperçu sur deux semaines exécuté : zéro résultat, montré sans inventer de correspondances |
| Weekly letters | [Lire](../guides/fr/weekly-science-letters.md) | [Read](../guides/en/weekly-science-letters.md) | Éditions Astronomie et espace du 26 octobre 2026, ouvertes dans les deux langues |
| Rising topics | [Lire](../guides/fr/rising-research-topics.md) | [Read](../guides/en/rising-research-topics.md) | Vue Premium Astrophysique, semaine du 12 octobre 2026, dans les deux langues |
| Scoop watch | [Lire](../guides/fr/scoop-watch-research-project.md) | [Read](../guides/en/scoop-watch-research-project.md) | Deux projets fictifs créés ; contrôle des annonces terminé ; recherche historique française terminée avec 67 articles proches, anglaise en échec après une relance |
| Citation alerts | [Lire](../guides/fr/track-research-paper-citations.md) | [Read](../guides/en/track-research-paper-citations.md) | Fiche d’un papier déjà suivi : 135 citations, analyse des contextes lancée dans les deux langues et extraits ouverts ; aucune nouvelle alerte email prétendument reçue |
| Who works on it? | [Lire](../guides/fr/find-research-teams-by-topic.md) | [Read](../guides/en/find-research-teams-by-topic.md) | Deux générations indépendantes terminées : 24 articles / 2 approches en français, 22 / 3 en anglais |
| Find a research placement | [Lire](../guides/fr/find-research-placement-contacts.md) | [Read](../guides/en/find-research-placement-contacts.md) | Deux recherches fictives L3/Bachelor, six mois, sans restriction géographique ; contacts obtenus et limites de couverture explicites |
| Press kit | [Lire](../guides/fr/research-paper-press-kit.md) | [Read](../guides/en/research-paper-press-kit.md) | Kits actualisés dans les deux langues sur 1703.10693v1 ; brief et parcours chercheur avec points forts et passages sources ; restriction à l’abstract indiquée |

## Objets de démonstration dans le compte

La connexion a été réalisée par l’utilisateur dans le navigateur intégré. Les formulaires payants ont ensuite été vérifiés par l’agent, sans changement de formule ni de droits.

- Collection 67 : « Guide démo — Gaz dans les disques de débris ». Les trois références réellement importées sont 1703.10693v1, 2110.02308v1 et 1606.01247v1. Les deux références initialement envisagées ont été remplacées lorsque leur import arXiv n’a pas abouti.
- Deux recherches d’équipes indépendantes ont été générées dans les deux langues. Le champ ORCID a été laissé vide pour ces exemples.
- Deux recherches de contacts ont été générées à partir d’un profil fictif ; aucune candidature ni demande de contact n’a été envoyée.
- Recherche sauvegardée 67 : `"debris discs" gas`. La veille par email n’a pas été activée.
- Projets Scoop Watch 34 et 35 : « Guide démo — Origine du gaz dans les disques de débris » et « Guide demo — Gas origins in debris discs ». Ils restent actifs : forts recouvrements immédiats, partiels hebdomadaires. Ils sont distincts du projet préexistant, qui n’a pas été modifié. Aucun manuscrit privé n’a été envoyé.
- L’abonnement au digest existant n’a pas été modifié. L’aperçu anglais utilise les réglages non enregistrés du formulaire. Les adresses privées signées des éditions et des flux ne sont pas reproduites dans les articles.
- Les liens publics de partage des conversations, collections et résultats n’ont pas été activés.

## Limites des démonstrations

Le projet Scoop Watch français a terminé sa recherche historique sur trois ans : 67 articles proches. Le projet anglais a échoué, y compris après une relance ; le guide anglais décrit honnêtement la configuration et le contrôle quotidien, puis renvoie au cas français pour les recouvrements historiques.

Plusieurs sources étaient indisponibles pendant la session ; les résultats incomplets, absents ou limités à l’abstract restent signalés. Une capture de « zéro résultat » ne prouve pas une absence de littérature.

Les lignes `Guide` en base prennent le pas sur les fichiers Markdown de même langue et slug. La publication repose sur les fichiers ; vérifier les 24 URLs anonymes après le déploiement avant de considérer la publication comme terminée. Aucun écrasement global des guides en base n’est nécessaire.

## Correctifs découverts en testant le chat

Le paramètre `?ask=` écrasait la fonction d’envoi des questions : nom de variable corrigé, tests avec et sans question préremplie. L’exemption de quota de l’administrateur renvoyait `1`, ce qui bloquait les tableaux de plusieurs papiers ; elle renvoie désormais une valeur illimitée, cohérente avec l’interface. Les règles des abonnements restent identiques. Les deux correctifs ont été déployés avant les nouvelles démonstrations.

## Vérifications

Les 24 guides sont rendus avec le template existant ; métadonnées, dates, descriptions, JSON-LD, images et liens internes sont contrôlés. Les nouvelles figures utilisent les dimensions réelles des JPEG et s’ouvrent en taille originale. Les 15 tests existants des guides passent. Les liens des 12 outils sont vérifiés en français et en anglais, sans lien dans les langues qui ne disposent pas de ces guides. Les captures des résultats ont été inspectées ; les titres des publications et les passages scientifiques cités gardent leur langue originale.

Les boutons des guides conservent leurs événements fixes `guide_tool_<slug>`, gérés par le script analytics existant. Aucune donnée de projet ou de profil n’est incluse dans ces événements. La collecte et l’attribution complète des conversions n’ont pas été vérifiées en production.

### Année des citations dans le chat

Les repères bibliographiques privilégient désormais l’année de publication issue du contrôle éditeur existant ou de la référence de revue ; l’année du dépôt sert de repli. Les papiers EHT I et VI portent donc les repères 2022a et 2022b. La date arXiv est explicitement nommée sur les fiches. Dix captures des comparaisons et tableaux ont été refaites dans les deux langues après cette correction, avec de nouvelles conversations privées. Les anciennes conversations conservent leurs réponses historiques.
