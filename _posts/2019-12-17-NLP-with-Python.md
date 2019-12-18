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

#### Minimum Widths, Alignment and Padding
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

### Working with PDF

> It's common to use **PyPDF2** to read the text. But it may not work because of the encoding issues or image.

```python

    import PyPDF2
    
    #First we open a pdf, then create a reader object for it. Notice how we use the binary method of reading , 'rb', instead of just 'r'.
    
    # Notice we read it as a binary with 'rb'
    f = open('US_Declaration.pdf','rb')
    
    pdf_reader = PyPDF2.PdfFileReader(f)
    
    pdf_reader.numPages
    page_one = pdf_reader.getPage(0)
    
    #We can then extract the text:
    page_one_text = page_one.extractText()
    
    f.close()
    
```

#### Adding to PDFs

```python
    f = open('US_Declaration.pdf','rb')
    pdf_reader = PyPDF2.PdfFileReader(f)
    
    first_page = pdf_reader.getPage(0)
    pdf_writer = PyPDF2.PdfFileWriter()
    pdf_writer.addPage(first_page)
    
    pdf_output = open("Some_New_Doc.pdf","wb")
    pdf_writer.write(pdf_output)
    
    pdf_output.close()
    f.close()
    
```

### Regular Expression

```python

    text = "The agent's phone number is 408-555-1234. Call soon!"
    import re
    pattern = 'phone'
    re.search(pattern,text)
    >>> <_sre.SRE_Match object; span=(12, 17), match='phone'>
    
    match = re.search(pattern,text)
    match.span()
    >>> (12, 17)
    match.start()
    >>> 12
    match.start()
    >>> 17
```

#### .findall()

```python
    # Notice it only matches the first instance. If we wanted a list of all matches, we can use .findall() method:
    matches = re.findall("phone",text)
    matches
    >>> ['phone', 'phone']
    len(matches)
    >>> 2
    
    for match in re.finditer("phone",text):
    print(match.span())
    >>> (3, 8)
(18, 23)

    #If you wanted the actual text that matched, you can use the .group() method.
    match.group()
    >>> 'phone'

```

#### Identifiers for Characters in Patterns

| Character | Description | Example Pattern Code | Exammple Match |
| :-----------: | :-----------: | :----------------: |:-----------: |
| \d    | A digit     | file_\d\d     |file_25     |
| \w    | Alphanumeric     | \w-\w\w\w     |A-b_1     |
| \s    | White space     | a\sb\sc     |a b c     |
| \D    | A non digit     | \D\D\D     |ABC     |
| \W    | Non-alphanumeric     | \W\W\W\W\W     |*-+=)     |
| \S    | Non-whitespace     | \S\S\S\S     |Yoyo    |

Example:

```python

    text = "My telephone number is 408-555-1234"
    phone = re.search(r'\d\d\d-\d\d\d-\d\d\d\d',text)
    phone.group()
    >>> '408-555-1234'

```

####  Quantifiers
> Notice the repetition of \d. That is a bit of an annoyance, especially if we are looking for very long strings of numbers. Let's explore the possible quantifiers.

![](https://i.imgur.com/biZn0LJ.png)

```python
    #use quantifier to match
    re.search(r'\d{3}-\d{3}-\d{4}',text)
```

#### Groups
> - What if we wanted to do two tasks, find phone numbers, but also be able to quickly extract their area code (the first three digits). We can use groups for any general task that involves grouping together regular expressions (so that we can later break them down). 
> - Using the phone number example, we can separate groups of regular expressions using parentheses:
```python
    phone_pattern = re.compile(r'(\d{3})-(\d{3})-(\d{4})')

    results = re.search(phone_pattern,text)

    # The entire result
    results.group()
    >>> '408-555-1234'
    
```

#### Additional Regex Syntax
- Or operator |

```python
    re.search(r"man|woman","This man was here.")
    >>> <_sre.SRE_Match object; span=(5, 8), match='man'>
    re.search(r"man|woman","This woman was here.")
    >>> <_sre.SRE_Match object; span=(5, 10), match='woman'>

```

#### The Wildcard Character
> Use a "wildcard" as a placement that will match any character placed there. You can use a simple period . for this. For example:

```python

    re.findall(r".at","The cat in the hat sat here.")

    #取字尾
    # One or more non-whitespace that ends with 'at'
    re.findall(r'\S+at',"The bat went splat")
    >>> ['bat', 'splat']
```

#### Starts With and Ends With
> We can use the ^ to signal starts with, and the $ to signal ends with:

```python

    # Ends with a number
    re.findall(r'\d$','This ends with a number 2')

    # Starts with a number
    re.findall(r'^\d','1 is the loneliest number.')
    
    #To get the each word
    phrase = "there are 3 numbers 34 inside 5 this sentence."
    re.findall(r'[^\d]+',phrase)
    >>> ['there are ', ' numbers ', ' inside ', ' this sentence.']
    
    #remove punctuation
    test_phrase = 'This is a string! But it has punctuation. How can we remove it?'
    
    re.findall('[^!.? ]+',test_phrase)
    >>> 
    ['This',
     'is',
     'a',
     'string',
     'But',
     'it',
     'has',
     'punctuation',
     'How',
     'can',
     'we',
     'remove',
     'it']

    clean = ' '.join(re.findall('[^!.? ]+',test_phrase))
    >>> 'This is a string But it has punctuation How can we remove it'
```

#### Brackets for Grouping
> As we showed above we can use brackets to group together options, for example if we wanted to find hyphenated words:

```python

    text = 'Only find the hypen-words in this sentence. But you do not know how long-ish they are'
    re.findall(r'[\w]+-[\w]+',text)
    >>> ['hypen-words', 'long-ish']

```

#### Parentheses for Multiple Options
> If we have multiple options for matching, we can use parentheses to list out these options. For Example:

```python
    # Find words that start with cat and end with one of these options: 'fish','nap', or 'claw'
    text = 'Hello, would you like some catfish?'
    texttwo = "Hello, would you like to take a catnap?"
    re.search(r'cat(fish|nap|claw)',text)
    >>> <_sre.SRE_Match object; span=(27, 34), match='catfish'>
    
    re.search(r'cat(fish|nap|claw)',texttwo)
    >>> <_sre.SRE_Match object; span=(32, 38), match='catnap'>

```
