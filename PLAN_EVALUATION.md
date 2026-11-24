# Livrable 2 — bilan et reste à faire (évaluation)

*État au mardi 24/11/2026 · rendu prévu le mercredi 25/11/2026*

Les notebooks 01 à 04 sont faits. Il reste **le notebook 05 (évaluation sur les ensembles de test)**,
puis les fichiers de gouvernance et une vérification finale.

---

## 1. Ce qui a été fait, et pourquoi

### Les notebooks

| Notebook | Ce qu'il fait | Pourquoi | Chiffres clés |
|---|---|---|---|
| `01_first_look` | Regarde un seul patient | Comprendre le format : 1 ligne = 1 heure, 41 colonnes | p000001 : 54 h, jamais septique |
| `02_data_preparation` | Charge les 40 336 fichiers, contrôles qualité, découpage par patient, variables de la baseline, test anti-fuite | Construire des données propres **sans fuite** | 1 552 210 heures ; A : 12 201 / 4 067 / 4 068 patients ; B : 12 000 / 4 000 / 4 000 ; test anti-fuite réussi sur 50 patients, version fausse détectée chez 43/50 |
| `03_eda` | Taille, déséquilibre, manquants, valeurs aberrantes, corrélations, différences A/B | Décrire avant de modéliser, mesurer le *shift* | 2,17 % d'heures positives en A, 1,41 % en B ; gaz du sang fréquents en A, quasi absents en B ; EtCO2 jamais mesuré en A |
| `04_baseline` | Classe majoritaire, règle qSOFA, régression logistique réglée et calibrée, sur A-train et A-calib | Avoir la référence à battre au livrable 3 | Logistique : AUPRC 0,092, AUROC 0,78 sur A-calib ; qSOFA : AUPRC 0,025, AUROC 0,56 |

### Les décisions prises

| Décision | Pourquoi |
|---|---|
| Découpage **par patient** 60/20/20 dans chaque hôpital, stratifié, graine 42 | Un patient ne doit jamais être à la fois dans le train et dans le test (fuite) |
| Les deux hôpitaux sont découpés | Le livrable 1 prévoit l'expérience dans les deux sens (A → B et B → A) |
| Trous remplis par la **dernière valeur connue du patient** + indicateurs « jamais mesuré » | N'utilise que le passé ; ne pas avoir mesuré un examen est une information |
| Aucune médiane, limite ou moyenne calculée hors du **Pipeline** | Tout ce qui est appris doit l'être sur le train uniquement |
| **AUPRC** comme métrique principale | Seulement ~2 % d'heures positives : toujours répondre « non » donne 98 % d'accuracy |
| Valeurs extrêmes **écrêtées** aux percentiles 0,5–99,5 % du train | Valeurs impossibles (50 °C, 300 mmHg) ; la logistique y est sensible |
| Unit2 et EtCO2 retirés | Unit2 = doublon de Unit1 ; EtCO2 n'est jamais mesuré en A |
| `StratifiedGroupKFold` (5 plis) pour le réglage | Groupé par patient (pas de fuite), et chaque pli garde la même part de positifs |
| **Pas de poids de classe** (`class_weight=None`) | Choisi par la validation croisée ; garde des probabilités bien calibrées. *Écart au livrable 1* |
| Calibration mesurée sur **10 groupes de même effectif** | 97,6 % des heures ont un risque prédit < 10 % : avec des tranches de même largeur, on ne voit rien |
| Correction de **Platt** gardée, ajustée sur A-calib | Inutile sur A, mais peut servir sur B, où la maladie est moins fréquente |

### Les fichiers produits (dans `data/processed/`, jamais sur GitHub)

- `hourly.pkl` — le tableau heure par heure (notebook 02)
- `patients_splits.csv` — chaque patient et son ensemble : train, calib ou test (notebook 02)
- `features_baseline.pkl` — les variables de la baseline (notebook 02)
- `baseline_logistic.joblib` — le modèle entraîné + la correction de Platt + la liste des colonnes (notebook 04)

### Ce qu'on a appris en chemin

- **Lactate** : sa valeur ne compte presque pas (+0,01), le fait de ne pas l'avoir dosé compte beaucoup (−0,20).
  Le modèle s'appuie sur la décision de prescrire l'examen plus que sur son résultat.
- **ICULOS** (temps passé en réanimation) est la variable la plus forte. Ce n'est pas une fuite, mais
  elle reflète peut-être la façon dont les organisateurs ont construit les séjours.
- **6 des 15 plus gros coefficients** sont des indicateurs « jamais mesuré », qui dépendent des habitudes
  de chaque hôpital : risque de mauvaise transposition à B.
- Les **10 % d'heures jugées les plus à risque** contiennent environ **43 % des heures septiques** (sur A-calib).

---

## 2. Ce qu'il reste : le notebook 05 (évaluation)

**But.** Mesurer **une seule fois** les modèles sur des patients jamais utilisés :
- **A-test** (4 068 patients) — évaluation *interne*, même hôpital ;
- **B entier** (20 000 patients) — évaluation *externe*, autre hôpital.

Puis chiffrer la perte entre les deux (objectif **O3** du livrable 1).

### Les étapes

| # | Étape | Pourquoi | À vérifier |
|---|---|---|---|
| 1 | Charger `features_baseline.pkl`, `patients_splits.csv` et `baseline_logistic.joblib` ; isoler A-calib, A-test et B | On **réutilise** le modèle du 04, sans rien réentraîner | Nombres de patients : 4 067, 4 068, 20 000 |
| 2 | Calculer les scores des 4 modèles (majorité, qSOFA, logistique brute, logistique + Platt) sur A-test et B, avec `predict_risk` | Même modèle, deux populations | Aucun `fit` sur A-test ni sur B |
| 3 | Fixer les **seuils d'alerte sur A-calib** pour un taux d'alerte de 5 % et 10 % des heures | Le seuil de 0,5 n'a pas de sens ici ; un seuil choisi sur les tests serait une fuite | Taux d'alerte réel sur A-calib ≈ 5 % / 10 % |
| 4 | Métriques sur chaque ensemble : AUPRC, AUROC, Brier, ECE, sensibilité au taux d'alerte fixé, **fausses alertes par patient-jour** | Métriques promises au livrable 1 | Classe majoritaire : AUROC 0,5 et AUPRC = part de positifs (sinon, bug) |
| 5 | **IC à 95 % par bootstrap sur les patients** (1 000 tirages) | Les heures d'un même patient se ressemblent : tirer des lignes donnerait des IC trop étroits | Chaque IC contient la valeur mesurée ; mêmes résultats en relançant (graine) |
| 6 | **Écarts interne − externe** (ΔAUROC, ΔAUPRC, ΔECE) avec leur IC | Objectif O3 | Signe : positif = moins bon sur B |
| 7 | **Diagrammes de fiabilité A-test vs B** (brut et Platt) | Voir si le risque prédit reste juste dans l'autre hôpital | Attendu : risque surestimé sur B (1,41 % de positifs contre 2,17 %) |
| 8 | Comparaison logistique vs qSOFA par **bootstrap apparié sur les patients**, avec correction de Holm | Remplace le test de DeLong, qui suppose des observations indépendantes. *Écart au livrable 1* | p-values non nulles |
| 9 | **Sens B → A** : même Pipeline entraîné sur B-train, Platt sur B-calib, évalué sur B-test et A entier | Engagement du livrable 1 | EtCO2 sera gardé (mesuré en B) mais toujours absent en A : à commenter |
| 10 | Figures pour le rapport : courbes ROC et PR, fiabilité, métriques avec IC | Le rapport s'appuie sur ces figures | Axes légendés, en anglais |
| 11 | Observations, avec les chiffres | Ce qu'on expliquera à l'oral | Chiffres identiques à ceux des sorties |

### Comment fonctionne le bootstrap par patient (étape 5)

1. On tire au hasard, **avec remise**, autant de patients qu'il y en a dans l'ensemble. Un patient peut
   être tiré plusieurs fois, un autre jamais.
2. On garde **toutes les heures** des patients tirés et on recalcule la métrique.
3. On recommence 1 000 fois. Les 2,5 % et 97,5 % des 1 000 valeurs obtenues forment l'IC à 95 %.

Pour un écart A-test − B, on fait la différence tirage par tirage : les deux ensembles sont
indépendants. B compte 762 000 heures : il faut un code efficace, ou un premier essai à 200 tirages
avant les 1 000 du résultat final.

### Ce à quoi on s'attend (à comparer avec les résultats)

- **AUROC et AUPRC plus basses sur B.** Les hôpitaux ne mesurent pas les mêmes choses, et le modèle
  s'appuie sur les indicateurs « jamais mesuré ».
- **L'AUPRC baisse aussi mécaniquement sur B**, car la maladie y est plus rare : un modèle au hasard y
  obtient 0,014 contre 0,022 en A. Il faut donc aussi comparer l'AUPRC **rapportée à la part de
  positifs** (combien de fois mieux que le hasard).
- **Risque surestimé sur B** : prévalence plus faible, et la correction de Platt, apprise sur A,
  ne la corrige pas.
- **Sur A-test**, des chiffres proches de ceux d'A-calib (AUPRC ~0,09, AUROC ~0,78). Sinon, chercher
  une erreur.

---

## 3. Après le 05 : finir le livrable 2

- [ ] Déposer les fichiers `.md` d'hier (`fichiers_md.zip`) : README complet, `CONTRIBUTIONS.md`,
      `PROMPT_LOG.md`, `AI_AUDIT.md`, README de `data/` corrigés
- [ ] `PROMPT_LOG.md` : au moins une entrée **écrite par chaque membre** (sinon −2 points individuels)
- [ ] `AI_AUDIT.md` : au moins une erreur d'IA **par membre** (candidates : la mémoire du notebook 02
      au premier lancement ; la mesure de la calibration en tranches de même largeur)
- [ ] `CONTRIBUTIONS.md` : qui présente quelle partie à l'oral
- [ ] README : cocher les notebooks 04 et 05
- [ ] Vérification finale : nouveau clone du repo, tous les notebooks relancés dans l'ordre, mêmes chiffres
- [ ] Rapport : plus tard

---

## 4. Pièges déjà rencontrés

- **Après une modification de `sepsis_utils.py`** : « Restart » avant « Run All », sinon Python
  garde l'ancienne version en mémoire.
- **Avant de pousser** : « Restart » + « Run All », puis vérifier qu'il n'y a pas d'erreur et que les
  chiffres sont les mêmes.
- **Avant de travailler** : `git pull`.
- **Jamais de `data/` sur GitHub** : `git status` ne doit montrer aucun `.pkl`, `.csv` ou `.psv` de `data/`.
- **Fichiers `.md`** : les ouvrir dans VS Code, pas dans le Bloc-notes, qui ajoute des `\`.
