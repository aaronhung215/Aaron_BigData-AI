---
layout: post
title: NLP - Natural Language Processing with Python [Udemy Course]
date: 2019-12-16
categories: Udemy Python NLP
comments: true
---
An Online course to learn how to use python to develop NLP.


## NLP 
        - Spacy Piplines
        - Tokenizations
        - Stemming
        - Lemmatizations
        - Stop Words
        - Vocabulary and Phrase Matching

## Parts of Speech Tagging
        - Parts-of-Speech(POS)
        - Named Entity Recognition(NER)
        - Sentence Segmentation with Spacy
        - Visualization of POS and NER

### 安裝環境

```bash
    conda env create -f nlp_course_env.yml
```

### 基本文字處理
#### 列印文字

```python
    person = "aaron"

    # Using the old .format() method:
    print("My name is {}".format(person))
    >> His name is aaron.

    # Using f-strings:
    print(f"My name is {person}")
    >> His name is aaron.


    d = {'a':123,'b':456}

    # Be careful not to let quotation marks 
    # in the replacement fields conflict 
    # with the quoting used in the outer string:
    print(f"Address: {d['a']} Main Street")
    >> Address: 123 Main Street
```

- Minimum Widths, Alignment and Padding
> You can pass arguments inside a nested set of curly braces to set a minimum width for the field, the alignment and even padding characters.

```python
    library = [('Author', 'Topic', 'Pages'), ('Twain', 'Rafting', 601), ('Feynman', 'Physics', 95), ('Hamilton', 'Mythology', 144)]

    for book in library:
        print(f'{book[0]:{10}} {book[1]:{8}} {book[2]:{7}}')
    >>>
    Author     Topic    Pages  
    Twain      Rafting      601
    Feynman    Physics       95
    Hamilton   Mythology     144    

```

> - Here the first three lines align, except Pages follows a default left-alignment while numbers are right-aligned. Also, the fourth line's page number is pushed to the right as Mythology exceeds the minimum field width of 8. When setting minimum field widths make sure to take the longest item into account.
> - To set the alignment, use the character < for left-align, ^ for center, > for right.
> - To set padding, precede the alignment character with the padding character (- and . are common choices).

```python
    for book in library:
        print(f'{book[0]:{10}} {book[1]:{10}} {book[2]:.>{7}}') # here .> was added

    >>>
    Author     Topic      ..Pages
    Twain      Rafting    ....601
    Feynman    Physics    .....95
    Hamilton   Mythology  ....144
```

#### Date Formating

```python
    from datetime import datetime

    today = datetime(year=2018, month=1, day=27)
    print(f'{today:%B %d, %Y}')

    >>> January 27, 2018
```


#### Ｗorking with Text file

```python
    #Create a file
    %%writefile test.txt
    Hello, this is a quick test file.
    This is the second line of the file.
    
    #append the content
    %%writefile -a test.txt
    This is more text being appended to test.txt
    And another line here.
    
    #Open the file
    my_file = open('test.txt')
    
    # We can now read the file
    my_file.read()
    >>> 'Hello, this is a quick test file.\nThis is the second line of the file.'
    
    # But what happens if we try to read it again?
    my_file.read()
    >>> ''
    
    # 指定文件讀取位置
    myfile.seek(0)
    >>> 0

    # Readlines returns a list of the lines in the file
    my_file.seek(0)
    my_file.readlines()
    >>> ['Hello, this is a quick test file.\n', 'This is the second line of the file.']
    
    my_file.close()
    
    
    # Add a second argument to the function, 'w' which stands for write.
# Passing 'w+' lets us read and write to the file
    #It will overwrite.
    my_file = open('test.txt','w+')
    
    #append
    my_file = open('test.txt','a+')
    my_file.write('\nThis line is being appended to test.txt')
    my_file.write('\nAnd another line here.')
```

##### Aliases and Context Managers

```python
    #You can assign temporary variable names as aliases, and manage the opening and closing of files automatically using a context manager:
    with open('test.txt','r') as txt:
        first_line = txt.readlines()[0]
    print(first_line)
    
    with open('test.txt','r') as txt:
    for line in txt:
        print(line, end='')  # the end='' argument removes extra linebreaks
```

