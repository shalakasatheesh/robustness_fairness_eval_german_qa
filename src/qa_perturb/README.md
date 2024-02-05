# Creating a new perturbation function

1. Copy and edit new_petrub.py to add a new perturbation
    1. Edit the name if the **file** to match the perturbation
    2. Edit the name of the **class** to match the perturbation
    3. Edit `self.name` to reflect the pertubation. Eg: `self.name: str = 'insert_chara_verb'`
    4. Edit docstrings to include details about the implementation.
    
2. Add name of new peturbation file to `__init__.py`. Eg:

        from qa_perturb.chara_perturb import InsertChara


# Currently implemented (Overview):

## Character Perturbations:

| <center> Package Name | <center> Details | <center> Example | 
| ------------- | --------| ----- | 
| [`DeleteChara`](chara_perturb/README.md#1-deletechara--insertchara--repeatchara--replacechara--swapchara) | Delete a random character (between the first and last characters) from a word from the `question` or the `context` | `wohnen` &rarr; `wonen` | 
| [`InsertChara`](chara_perturb/README.md#1-deletechara--insertchara--repeatchara--replacechara--swapchara) | Insert a random alphabet (between the first and last characters) from a word from the `question` or the `context` | `wohnen` &rarr; `wobhnen` |  
| [`RepeatChara`](chara_perturb/README.md#1-deletechara--insertchara--repeatchara--replacechara--swapchara) | Repeat a random character (between the first and last characters) from a word from the `question` or the `context` | `wohnen` &rarr; `woohnen` |  
| [`ReplaceChara`](chara_perturb/README.md#1-deletechara--insertchara--repeatchara--replacechara--swapchara) | Delete a random character and replace with a random character (between the first and last characters) from a word from the `question` or the `context` | `wohnen` &rarr; `wobnen` |  
| [`SwapChara`](chara_perturb/README.md#1-deletechara--insertchara--repeatchara--replacechara--swapchara) | Swap two random characters from a word from the `question` or the `context` | `wohnen` &rarr; `whonen` |  
| [`KeyboardTypo`](chara_perturb/README.md/#5-keyboardtypo) | Choose and random word and produce a typo depending on the keyboard layout | `welcher` &rarr; `welyher` | 
| [`DeletePunctuation`](chara_perturb/README.md#2-deletepunctuation) | Deletes all punctuations from the `question` or the `context` | `Was kann den Verschleiß des seillosen Aufzuges minimieren? ` &rarr; `Was kann den Verschleiß des seillosen Aufzuges minimieren ` |
| [`InsertPunctuation`](chara_perturb/README.md#6-insertpunctuation) | Insert punctuations at random positions for a randomly chosen word from the `question` or the `context` | `welcher` &rarr; `welche/r` |
| [`ChangeCase`](chara_perturb/README.md#3-changecase) | Change/Invert the case of the `question` or `context` to lower, upper or title. | `In welcher deutschen Stadt wird der seillose Aufzug getestet?` &rarr; `IN WELCHER DEUTSCHEN STADT WIRD DER SEILLOSE AUFZUG GETESTET?` |
| [`ReplaceUmlaute`](chara_perturb/README.md#4-replaceumlaute) | Replace umlaute with the following translation table `{'ä':'ae', 'Ä':'AE', 'ü':'ue', 'Ü':'UE', 'ö':'oe', 'Ö':'OE', 'ß':'ss'}` | `ausgewählt` &rarr; `ausgewaehlt` |
Insert Number, Convert number to text, Insert cont. word.

## Word Perturbations:
| <center> Package Name | <center> Details | <center> Example | 
| ------------- | --------| ----- | 
| [`DeleteWord`](word_perturb/README.md#1-deleteword--repeatword--splitword) | **Might alter sematics**. Delete a random word from the `'question'` or the `'context'`. | `In welcher deutschen Stadt wird der seillose Aufzug getestet?` &rarr; `In deutschen Stadt wird der seillose Aufzug getestet?` |  
| [`RepeatWord`](word_perturb/README.md#1-deleteword--repeatword--splitword) | Repeat a random word from the `'question'` or the `'context'`. | `In welcher deutschen Stadt wird der seillose Aufzug getestet?` &rarr; `In welcher welcher deutschen Stadt wird der seillose Aufzug getestet?` |  
| [`Synonym`](word_perturb/README.md#2-synonym) | Replaces a verb from the `'question'` or the `'context'` with it's contextual synonym. CURRENT IMPLEMENTATION LIMITATIONS: (1) only works for `pos_tag=verb`(2) does not work as well for `'context'` because of model's limit to sequence length. | `Was kann den Verschleiß des seillosen Aufzuges minimieren?` &rarr; `Was kann den Verschleiß des seillosen Aufzuges verhindern?` |  
| [`SplitWord`](word_perturb/README.md#1-deleteword--repeatword--splitword) | Add a space in a randomly chosen word from the `'question'` or the `'context'`. | `Aufzug` &rarr; `Auf zug` | 
| [`SwapWords`](word_perturb/README.md#3-swapwords) | Swaps two random words in a sentence from the `'question'` or the '`context`'. | `In welcher deutschen Stadt wird der seillose Aufzug getestet?` &rarr; `In welcher welcher Stadt deutschen wird der seillose Aufzug getestet?` | 

## Sentence Perturbations:
| <center> Package Name | <center> Details | <center> Example |
| ------------- | --------| ----- | 
| [`BackTranslate`](sentence_perturb/README.md#1-backtranslate) | Translate the `'question'` from German to English and then back to German.  `'context'` translation is currently not supported due to quality and length constraints. |  `Was kann den Verschleiß des seillosen Aufzuges minimieren?` &rarr; `Wie kann der Verschleiß des drahtlosen Aufzugs minimiert werden?` | 
| [`RepeatSentence`](sentence_perturb/README.md#2-repeatsentence) | Repeat the `'question'` or the `'context'` | `Was kann den Verschleiß des seillosen Aufzuges minimieren?` &rarr; `Was kann den Verschleiß des seillosen Aufzuges minimieren? Was kann den Verschleiß des seillosen Aufzuges minimieren?` |  
insert a cont. sentence as a suggestion from an LLM -> distraction sentence