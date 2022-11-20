## Ablation study for BG's slice embedding method
||F1|P|R|ACC|
|-|-|-|-|-|
|BG with Word2vec|54.7|52.0|56.7|56.4|
|BG with CodeBERT|56.1|51.8|61.2|56.4|

## Performance of CodeBERT Classifier with different max sequence lengths
|seq len|F1|P|R|ACC|
|-|-|-|-|-|
|64|45.7|54.6|39.4|57.5|
|128|48.3|54.3|43.5|57.6|
|192|51.2|54.9|47.9|58.3|
|256|51.0|53.7|48.5|57.5|
|320|51.4|53.1|49.7|57.1|
|384|51.3|54.3|48.6|57.9|
|448|53.0|55.4|50.7|58.9|
|512|53.2|56.1|50.6|59.5|