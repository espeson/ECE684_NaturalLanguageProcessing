# POS Tagger Performance Analysis

## Word-by-Word Comparison Against Truth

**Overall Accuracy**: 42/45 words correct (93.3%)

### Sentence 10150 Analysis
| Word | True Tag | Predicted Tag | Correct? | Analysis |
|------|----------|---------------|----------|----------|
| Those | DET | DET | ✓ |  |
| **coming** | **VERB** | **NOUN** | ✗ | **Error**: Participle vs. gerund ambiguity |
| from | ADP | ADP | ✓ |  |
| other | ADJ | ADJ | ✓ |  |
| denominations | NOUN | NOUN | ✓ |  |
| will | VERB | VERB | ✓ |  |
| welcome | VERB | VERB | ✓ |  |
| the | DET | DET | ✓ | |
| opportunity | NOUN | NOUN | ✓ |  |
| to | PRT | PRT | ✓ |  |
| become | VERB | VERB | ✓ | |
| informed | VERB | VERB | ✓ |  |
| . | . | . | ✓ |  |

### Sentence 10151 Analysis
| Word | True Tag | Predicted Tag | Correct? | Analysis |
|------|----------|---------------|----------|----------|
| The | DET | DET | ✓ |  |
| preparatory | ADJ | ADJ | ✓ |  |
| class | NOUN | NOUN | ✓ | |
| is | VERB | VERB | ✓ | |
| an | DET | DET | ✓ |  |
| introductory | ADJ | ADJ | ✓ | |
| **face-to-face** | **ADJ** | **NOUN** | ✗ | **Error**: Compound adjective not recognized |
| group | NOUN | NOUN | ✓ | |
| in | ADP | ADP | ✓ | |
| which | DET | DET | ✓ |  |
| new | ADJ | ADJ | ✓ | |
| members | NOUN | NOUN | ✓ |  |
| become | VERB | VERB | ✓ |  |
| acquainted | VERB | VERB | ✓ |  |
| with | ADP | ADP | ✓ | |
| one | NUM | NUM | ✓ | |
| **another** | **DET** | **NOUN** | ✗ | **Error**: Indefinite determiner vs. pronoun confusion |
| . | . | . | ✓ | |

### Sentence 10152 Analysis
| Word | True Tag | Predicted Tag | Correct? | Analysis |
|------|----------|---------------|----------|----------|
| It | PRON | PRON | ✓ |  |
| provides | VERB | VERB | ✓ |  |
| a | DET | DET | ✓ |  |
| natural | ADJ | ADJ | ✓ |  |
| transition | NOUN | NOUN | ✓ |  |
| into | ADP | ADP | ✓ |  |
| the | DET | DET | ✓ |  |
| life | NOUN | NOUN | ✓ |  |
| of | ADP | ADP | ✓ |  |
| the | DET | DET | ✓ |  |
| local | ADJ | ADJ | ✓ |  |
| church | NOUN | NOUN | ✓ |  |
| and | CONJ | CONJ | ✓ |  |
| its | DET | DET | ✓ |  |
| organizations | NOUN | NOUN | ✓ |  |
| . | . | . | ✓ |  |


## Why It Produces Correct Tags

- **Large training data** (10k sentences) provides good word-tag associations
- **Add-1 smoothing** prevents zero probabilities for unseen transitions
- **Viterbi algorithm** finds optimal tag sequence using both word probabilities and tag transitions
- **Common patterns** like DET→NOUN, VERB→NOUN are well-learned

## Why It Makes Errors

**"coming" (VERB → NOUN)**
- Word can be both verb (participle) and noun (gerund)
- Model learned stronger noun association from training data

**"face-to-face" (ADJ → NOUN)**  
- Compound adjective not recognized as single unit
- Model treats hyphenated words as separate tokens

**"another" (DET → NOUN)**
- Word functions as both determiner and pronoun
- Training data bias toward noun classification

## Main Limitations

- **Limited context**: Only considers previous tag, not full sentence structure
- **No morphology**: Cannot analyze word formation or compounds
- **Training bias**: Reflects frequency patterns from Brown corpus
