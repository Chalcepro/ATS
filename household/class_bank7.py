"""Class lessons, batch 8 (11-02): feelings said straight to Coal, the way people really say them.

The older feelings lessons were framed ("Someone says: I'm sad. What do you say back?"); in a chat
people just say "i am sad today". Each feeling: four plain ways of saying it (the last held back)
and a kind, short answer. Same shape as class_bank.py.
"""

FEEL = [
    ("sad", ["I'm sad.", "I am sad today.", "I feel sad.", "Today I just feel so sad."],
     "i'm sorry you feel sad. do you want to talk about it? i'm here"),
    ("down", ["I'm feeling down.", "I feel low today.", "I'm not in a good mood.", "Honestly I'm a bit down right now."],
     "i'm sorry. what's making you feel down?"),
    ("happy", ["I'm happy!", "I feel really happy today.", "I'm in a great mood.", "Today I am so happy."],
     "that's great! what made you happy?"),
    ("excited", ["I'm so excited!", "I can't wait!", "I'm really excited about this.", "This has me so excited."],
     "nice! what are you excited about?"),
    ("tired", ["I'm tired.", "I feel so tired today.", "I'm exhausted.", "Man, I am really tired right now."],
     "you should rest a bit. i'll be here when you're back"),
    ("angry", ["I'm angry.", "I'm so mad right now.", "This makes me angry.", "I feel really angry today."],
     "i'm sorry. take a deep breath. what happened?"),
    ("scared", ["I'm scared.", "I feel afraid.", "This scares me.", "I'm a bit scared right now."],
     "it's okay, you're not alone. what are you scared of?"),
    ("worried", ["I'm worried.", "I'm worried about something.", "I can't stop worrying.", "Something is worrying me today."],
     "i'm sorry. tell me what's worrying you, maybe we can sort it out"),
    ("stressed", ["I'm stressed.", "I'm so stressed out.", "There's too much to do and I'm stressed.", "I feel really stressed today."],
     "let's take it one thing at a time. what's first?"),
    ("lonely", ["I'm lonely.", "I feel alone.", "Nobody is around and I feel lonely.", "I feel so lonely today."],
     "i'm here with you. do you want to talk?"),
    ("bored", ["I'm bored.", "I'm so bored.", "There's nothing to do.", "I'm really bored right now."],
     "want to play a game or learn something new?"),
    ("sick", ["I'm sick.", "I don't feel well.", "I feel ill today.", "I think I'm getting sick."],
     "i'm sorry. rest, drink water, and see a doctor if it gets worse"),
    ("hurt", ["I got hurt.", "I hurt myself.", "I'm in pain.", "Ouch, I just hurt my hand."],
     "are you okay? clean it and rest it, and get help if it's bad"),
    ("confused", ["I'm confused.", "I don't understand.", "This is confusing me.", "I'm really lost on this."],
     "that's okay. tell me which part and we'll go through it"),
    ("nervous", ["I'm nervous.", "I feel nervous about tomorrow.", "My nerves are bad today.", "I'm so nervous right now."],
     "that's normal. you've got this. what's coming up?"),
    ("proud", ["I'm proud of myself.", "I did it!", "I finally finished it.", "I feel proud today."],
     "well done! you should be proud"),
    ("grateful", ["Thank you so much.", "I'm grateful for your help.", "I really appreciate it.", "Thanks a lot for that."],
     "you're welcome, any time"),
    ("frustrated", ["I'm frustrated.", "Nothing is working.", "This is so annoying.", "I'm really frustrated right now."],
     "i'm sorry. let's look at it together, step by step"),
    ("calm", ["I feel calm.", "I'm relaxed.", "I feel peaceful today.", "Right now I'm really chill."],
     "that's good. enjoy it"),
    ("hungry", ["I'm hungry.", "I'm starving.", "I need food.", "I'm so hungry right now."],
     "let's get you something to eat"),
    ("lost a game", ["I lost the game.", "I keep losing this level.", "We lost the match.", "I lost again, ugh."],
     "that's okay, try again. you'll get better each time"),
    ("won a game", ["I won the game!", "I beat the level!", "We won the match!", "I finally won!"],
     "nice one! well played"),
]

MORE8 = [("feelings", asks, answer) for _, asks, answer in FEEL]
