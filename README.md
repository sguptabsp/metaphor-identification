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


### The objects

Different objects are used in this project. This section describes the different classes of objects and how to use them.

##### The Annotator
Defined in _/sample/modules/annotator.py_.



##### The Annotated Text
Defined in _/sample/modules/datastructs/annotated_text.py_.

##### The Identifier
Defined in _/sample/modules/cand_id.py_.

##### The Candidates
Defined in _/sample/modules/datastructs/candidate_group.py_.

The Candidates are actually a group of candidates. The class Candidate is defined in _/sample/modules/datastructs/metaphor_candidate.py_.

##### The Labeler
The Labeler is actually of the class MetaphorIdentifier defined in _/samples/modules/met_id.py_.

##### The Results
The results are returned from the metaphor-identifying method.

They usually follow this format:
* Source: concept
* Target: word qualified by the source
* Result: boolean
* Confidence: confidence measure between 0 and 1 
 

# Tutorial

#### The number of '#' defines the size of the heading

You can write words in **bold**, in _italic_ or **_both_**.