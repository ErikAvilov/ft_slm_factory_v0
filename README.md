# ft_slm_factory_v0

Fine Tuning d'un SLM avec Unsloth et LoRA.

## Outils pertinents :

[Datasets](https://huggingface.co/docs/datasets/index): Sert à télécharger un dataset depuis huggingface, dans notre cas [Banking77](https://huggingface.co/datasets/PolyAI/banking77).
#### Un dataset est un ensemble de données pouvant être utilisé pour entraîner une IA.

[Transformers](https://huggingface.co/docs/transformers/index): Aide à charger et utiliser un LM (language model).<br>
Ici je m'en sers pour télécharger **Qwen2.5-0.5B-Instruct**, charger son tokenizer, initialiser le modèle pour l'entraînement et faire les prédictions.
#### Le tokenizer est un outil permettant de transformer le texte en une représentation numérique compréhensible par un modèle.

[PEFT](https://huggingface.co/docs/peft/index) **(Parameter-Efficient Fine-Tuning)**: La librairie permettant d'appliquer des techniques de fine-tuning en paramètres, notamment LoRA. <br>
Concrètement, au lieu de réentraîner un modèle d'origine, LoRA le fige et ajoute de petites quantités de nouveaux paramètres, évitant la surcharge de ressource nécessaire dans le cas contraire.<br>
Mon PC personnel n'étant pas une machine de guerre, c'est l'outil parfait.

[TRL](https://huggingface.co/docs/trl/index) **(Transformers Reinforcement Learning)**: Fournit un ensemble d'outils spécialisés pour les LM. <br>
Dans ce projet, j'utilise notamment SFTTrainer **(Supervised Fine-Tuning Trainer)** pour entraîner le modèle à partir de dataset où la bonne réponse est déjà connue. <br>
Cela évite d'avoir à coder manuellement toute une boucle d'entraînement avec PyTorch.

[scikit-learn](https://scikit-learn.org/stable/): Spécialement prévu pour tester notre modèle entraîné, scikit-learn va comparer la liste des prédictions à la liste des vraies réponses pour calculer la précision ou le F1-score.

#### Le F1-score ou la [F-mesure](https://fr.wikipedia.org/wiki/F-mesure) est la mesure de performance d'un modèle d'intelligence artificielle en combinant sa précision et son rappel.
