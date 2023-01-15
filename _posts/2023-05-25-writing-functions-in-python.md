---
layout: post
title: Writing Functions in Python
date: 2023-01-15
tags: python
categories: python
comments: true
---

Writing Functions in Python

---

## Docstrings

1. Googlestyle
2. Numpydoc

---

Example
```python

def count_letter(content, letter):
  """Count the number of times `letter` appears in `content`.

  Args:
    content (str): The string to search.
    letter (str): The letter to search for.

  Returns:
    int

  # Add a section detailing what errors might be raised
  Raises:
    ValuesError: If `letter` is not a one-character string.
  """
  if (not isinstance(letter, str)) or len(letter) != 1:
    raise ValueError('`letter` must be a single character string.')
  return len([char for char in content if char == letter])



```


---

1. way1
`print(Function.__doc__)`
3. way2

```python
import inspect
print(inspect.getdoc(function)) ## d
```



---

## Dry and Do one thing

---

### Dry 
![](https://i.imgur.com/3RptgoN.png)

---

### Do One thing
![](https://i.imgur.com/oQs7ioC.png)


---

### Improvement
![](https://i.imgur.com/uXbe6Da.png)

---

## Context Manager

---

### Writing 

```python=

defcopy(src, dst):
    """Copy the contents of one file to another.  
    
    Args:
        src (str): File name of the file to be copied.    
        dst (str): Where to write the new file.  
    """
    # Open both files
    with open(src) as f_src:
        with open(dst, 'w') as f_dst:
        # Read and write each line, one at a time
        for line in f_src:        
            f_dst.write(line)



```

---

### Handle Errors


1. try: code that might raise an error
2. except: do something about the error
4. finally:this code runs no matter what


---

```python

def in_dir(directory):
  """Change current working directory to `directory`,
  allow the user to run some code, and change back.

  Args:
    directory (str): The path to a directory to work in.
  """
  current_dir = os.getcwd()
  os.chdir(directory)

  # Add code that lets you handle errors
  try:
    yield
  # Ensure the directory is reset,
  # whether there was an error or not
  finally:
    os.chdir(current_dir)


```

---


### Useful for context manager

![](https://i.imgur.com/QYDaDYz.png)


---

## Decorators


---

![](https://i.imgur.com/iUv88rS.png)


---


![](https://i.imgur.com/QehlP8m.png)


---


### Timer

![](https://i.imgur.com/sRgiuvR.png)



---

### Timeout

![](https://i.imgur.com/xOn8qHm.png)


