# Metaphors



### Command Line
When executing the program it is possible to use different arguments:
* -v or --verbose: print the details
* -a or --all: use every available methods of identification and labeling
* -m or --method followed by a number: choose the metaphor identifying method:
    * 1: Darkthought
    * 2: Clustering
* -i or --id followed by a number: choose the candidate identifying method:
    * 1: AdjNoun
    * 2: VerbNoun
* -f or --file followed by a file name: look for metaphors in a text file
* -s or --string followed by a string: look for metaphors in a specified string

If no string or text file is specified in the command line then a default text is used as an example.

<br>

### The execution

1. Tokenization of the text
    1. From string to annotatedText
    2. Identification of the part-of-speech of the words
2. Candidate identification
    * _What is this step?_
3. Metaphor identification
    * _What is this step?_

<br>

### The objects

Different objects are used in this project. This section describes the different classes of objects and how to use them.

#### The Annotator
Defined in _/sample/modules/annotator.py_.

It has 2 fields:
* rawText: string, text you want to analyze
* annotatedText: object of class annotatedText. See details [below](#annotatedTextAnchor).

#### <a name="annotatedTextAnchor">The Annotated Text</a>
Defined in _/sample/modules/datastructs/annotated_text.py_.

It has 5 fields:
* rawText
* table
* words
* size
* columns

Printing the annotated text will only print the table field.

#### The Identifier
Defined in _/sample/modules/cand_id.py_.

It has 2 fields:
* annotatedText: object of class annotatedText. See details [above](#annotatedTextAnchor).
* candidates: object of class CandidateGroup. See details [below](#candidatesAnchor)

#### <a name="candidatesAnchor">The Candidates</a>
Defined in _/sample/modules/datastructs/candidate_group.py_.

It has 2 fields:
* candidates: list of objects of class Candidate.
* size: number of candidates in the list candidates.

The class Candidate is defined in _/sample/modules/datastructs/metaphor_candidate.py_.
It has ... fields:
* annotatedText: object of class annotatedText. See details [above](#annotatedTextAnchor).
* sourceIndex:
* sourceSpan:
* targetIndex:
* targetSpan:

#### The Labeler
The Labeler is actually of the class MetaphorIdentifier defined in _/samples/modules/met_id.py_.

It has 2 fields:
* candidates: object of class candidates. See details [above](#candidatesAnchor)
* metaphors

#### The Results
The results correspond to the field metaphors of the labeler.
They are returned from the metaphor-identifying method.

They usually follow this format:
* Source: concept
* Target: word qualified by the source
* Result: boolean
* Confidence: confidence measure between 0 and 1 

<br> 

# Tutorial

#### The number of '#' defines the size of the heading

You can write words in **bold**, in _italic_ or **_both_**.