#!/usr/bin/env python

import os
import numpy as np
import pandas as pd

os.chdir('/Users/power/code/flashcards')

INPUT_CSV = 'wk20.csv'

PREFIX = r'''
\begin{filecontents}{avery83.cfg}
\newcommand{\cardpapermode}{portrait}
\newcommand{\cardpaper}{letterpaper}
\newcommand{\cardrows}{8}
\newcommand{\cardcolumns}{3}
\setlength{\cardheight}{1.3in}
\setlength{\cardwidth}{2.766in}
\setlength{\topoffset}{0.2in}
\setlength{\oddoffset}{0.1in}
\setlength{\evenoffset}{0.1in}
\end{filecontents}

\documentclass[avery83,grid]{flashcards}

\usepackage{xeCJK}
\setmainfont{Noto Serif Light}
\setsansfont{Noto Sans}
\setmonofont[Scale=MatchLowercase]{Noto Mono}

\setCJKmainfont{Noto Serif CJK JP}
\setCJKsansfont{Noto Sans CJK JP}
\setCJKmonofont{Noto Sans Mono CJK JP}

\usepackage{fontspec}
\usepackage{anyfontsize}
\fboxsep=0pt

\cardfrontstyle{headings}
\begin{document}
'''

SUFFIX = r'''
\end{document}
'''

SKIP_JLPT = {1,}

def load_jlpt():
  vocab_to_level = {}
  for filename in ['n1.csv', 'n2.csv', 'n3.csv', 'n4.csv', 'n5.csv']:
    df = pd.read_csv(f'data/{filename}', sep=',', names=['level', 'vocab'])
    for _, row in df.iterrows():
      vocab_to_level[row['vocab']] = max(
        row['level'], 
        vocab_to_level.get(row['vocab'], -1))
  return vocab_to_level


def load_frequencies():
  frequencies = {}
  df = pd.read_csv('data/frequency.csv', sep=',')
  for _, row in df.iterrows():
    frequencies[row['lemma']] = row['frequency']
  return frequencies


def load_genki():
  df = pd.read_csv('data/genki.txt', sep='\t', names=[
    'jp', 'en', 'group', 'lesson', 'part', 'type'
  ])
  seen = set()
  rows = []
  for _, row in df.iterrows():
    if '<ruby>' in row['jp']:
      # row contains both kanji and hiragana: <ruby>kanji<rt>hiragana</rt></ruby>
      # select out the kanji and hiragana pieces with a regex
      import re
      match = re.search(r'<ruby>(.*)<rt>(.*)</rt></ruby>', row['jp'])
      kanji = match.group(1)
      hiragana = match.group(2)
    else:
      kanji = row['jp']
      hiragana = row['jp']
    
    if not hiragana in seen:
      rows.append({
        'Item': kanji,
        'Reading': hiragana,
        'Meaning': row['en'],
      })
      seen.add(hiragana)
  
  df = pd.DataFrame(rows)
  return df

def load_wanikani():
  kept = []
  for level in [10, 20, 30, 40, 50]:
    df = pd.read_csv(f'data/wk{level}.csv', sep=';', na_values=[''], keep_default_na=False)
    readings = df['Reading'].unique()

    # assemble unique rows with the shortest context sentences
    for reading in readings:
      rows = df[df['Reading'] == reading]
      shortest = rows.iloc[0]
      for _, row in rows.iterrows():
        if len(row['Context Sentence JP']) < len(shortest['Context Sentence JP']):
          shortest = row
      kept.append({
        'Item': shortest.get('Item'),
        'Reading': shortest['Reading'],
        'Meaning': shortest['Meaning'],
        'Level': level,
        'Context Sentence JP': shortest['Context Sentence JP'],
        'Context Sentence EN': shortest['Context Sentence EN'],
      })

  return pd.DataFrame(kept)

def write_cards(df):
  total_cards = 0
  os.makedirs('out/', exist_ok=True)
  with open('out/flashcards.tex', 'w') as f:
    f.write(PREFIX)
    for _, row in df.iterrows():
      item, reading, meaning, context_jp, context_en = (
        row.get('Item', None), 
        row['Reading'], 
        row['Meaning'], 
        row.get('Context Sentence JP', ''),
        row.get('Context Sentence EN', '')
      )
      jlpt = jlpt_levels.get(item, -1)

      if not item in frequencies and jlpt in SKIP_JLPT:
        print(f'Skipping {item} / {meaning} because it is not in the frequency list')
        continue

      # don't keep context sentences that are too long
      if len(context_en.split(' ')) > 10:
        # print(f'Omitted context: {context_en}')
        context_en = context_jp = ''

      context_en = context_en.replace("’", "'")
      context_jp = context_jp.replace("’", "'")

      total_cards += 1

      f.write(rf'''
  \begin{{flashcard}}[{jlpt}]
  {{{reading}
  \vfill
  {context_jp}}}
  {meaning}
  \vfill
  {context_en}
  \end{{flashcard}}
  ''')
    f.write(SUFFIX)

  print('Total cards:', total_cards, ' pages: ', total_cards / 24 * 2)

genki = load_genki()
jlpt_levels = load_jlpt()
frequencies = load_frequencies()
wanikani = load_wanikani()

print(genki.head())
print(len(jlpt_levels))
print(len(frequencies))

wk_readings = {}
wk_items = {}
for _, row in wanikani.iterrows():
  wk_readings[row['Reading']] = row
  wk_items[row['Item']] = row

genki_kept = []
for _, row in genki.iterrows():
  item = row['Item']
  reading = row['Reading']
  if item in wk_items and wk_items[item]['Level'] == 10:
    print(f'Skipping: {item} because it was in WK10')
  elif reading in wk_readings and wk_readings[reading]['Level'] == 10:
    print(f'Skipping: {item} because it was in WK10')
  else:
    genki_kept.append(row)

genki = pd.DataFrame(genki_kept)
print(genki.head())
write_cards(genki)