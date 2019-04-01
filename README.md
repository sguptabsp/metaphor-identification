# Metaphors

### main.py

The file main.py represents the process of finding metaphors in a text:
1. Text Segmentation
2. Finding candidates for the metaphors: a candidate is a pair of word: adjective-noun or verb-noun 
3. Labeling the metaphors: Is a candidate metaphorical or literal? 

### Command Line
Mandatory arguments:
* -ml or --labeler followed by an ID: choose the metaphor labeling method:
    * darkthoughts
    * cluster
* -cf or --finder followed by an ID: choose the candidate finding method:
    * adjNoun
    * verbNoun
    
Optional  arguments:
* -v or --verbose: print the details
* -a or --all: use every available methods of identification and labeling
* -f or --file followed by a file name: look for metaphors in a text file
* -s or --string followed by a string: look for metaphors in a specified string

If no string or text file is specified in the command line then a default text is used.

### The Execution
#### Initialization

1. Parsing the command line
2. Creating a hash-table
    1. Adding the candidate finder functions
    2. Adding the metaphor labeler functions
3. Initializing the text either from:
    1. Default text - defined in _modules/utils.py_
    2. A string - written in the command line
    3. A file - path in the command line
4. Creating the object MetaphorIdentification
    
#### Step 1: Text Segmentation


#### Step 2: Finding Candidates
Call the procedure _MetaphorIdentification.findCandidates()_

#### Step 3: Labeling Metaphors
Call the procedure _MetaphorIdentification.labelMetaphors()_
    
### The Registry Class
Defined in _/new\_structure/modules/datastructs/registry.py_.

To identify metaphors in a text, at least two steps need to be followed:
the candidate identification step and the labelling step.
Each of these steps can be done in many ways.
Each method needs to be registered in the metaphorRegistry defined in 
_/sample/modules/registry.py_

### The MetaphorIdentification Class
Defined in _/new\_structure/modules/datastructs/MetaphorIdentification.py_.

It has four fields:
* rawText: string
* annotatedText: class AnnotatedText from _modules/datastructs/annotated_text.py_
* candidates: class CandidateGroup from _modules/datastructs/candidate_group.py_
* metaphors: class MetaphorGroup from _modules/datastructs/labeled_metaphor_list.py_



# Tutorial

#### The number of '#' defines the size of the heading

You can write words in **bold**, in _italic_ or **_both_**.