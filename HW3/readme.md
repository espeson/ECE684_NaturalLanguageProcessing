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


## Error Analysis

### Why Correct Tags Are Produced:
1. **Frequent Words**: Common words like "the", "is", "and" have clear, unambiguous patterns
2. **Strong Context**: DET→NOUN, VERB→ADV sequences are well-learned from training data
3. **Syntactic Patterns**: The model captures typical English word order effectively

### Why Errors Occur:

**"coming" (VERB → NOUN)**:
- **Issue**: Can function as present participle (VERB) or gerund (NOUN)
- **Context**: "Those coming from..." - participle modifying "Those"
- **Model Error**: Training data likely shows more NOUN usage for "-ing" forms

**"face-to-face" (ADJ → NOUN)**:
- **Issue**: Compound adjective treated as separate tokens
- **Context**: Should modify "group" as single adjectival unit
- **Model Error**: Lacks morphological awareness for hyphenated compounds

**"another" (DET → NOUN)**:
- **Issue**: Functions as both determiner and indefinite pronoun
- **Context**: "with one another" - "another" is determiner here
- **Model Error**: May have stronger NOUN association in training data

The tagger's 93.3% accuracy demonstrates effective learning of statistical patterns, but linguistic complexity in ambiguous cases remains challenging for the HMM approach.