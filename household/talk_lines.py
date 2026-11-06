"""Chalce's own lines (typed 11-06, Downloads/words.txt), grouped by what he means, plus wordings added around them.

His lines are kept as he typed them, except clear typos ("do i that way" -> "do it that way", "an we" -> "and we").
Stretched letters ("waaaiiiiit", "STOOOOP") are squashed by understand.words(), so they need no copies here.
Every fifth of HIS lines in a group is held back for the exam (never taught), so the exam asks: does Coal get what
Chalce means in words of Chalce's he never saw?
"""

MINE = {
    "greet": ["yooo what's good", "yo what's gud", "yo", "hey", "heylooo", "hellooo", "heloo", "heyo"],
    "help": ["i need your help with this file", "i need your help with a file", "i need help on a project",
             "i need help in sorting something", "is it possible for you to help me", "is it possible to solve this",
             "can you help me with this", "please there's something i need your help with",
             "please i need you to solve this", "please i need you to edit this", "can you edit this",
             "btw can you edit this", "btw can you solve this", "btw can you fix this",
             "is there anything you can do to solve this", "please i need your help with something",
             "please i need your help"],
    "can_you": ["okay so you can fix it right", "it's solvable", "what can be done here", "oh wait you can solve it",
                "shouldn't it be something you could solve", "well im sure you can do it"],
    "stop": ["noooo wait", "wait", "wait give me a sec", "noo err hmm wait", "give me a sec", "pause", "PAUSE",
             "STOOOOP", "stop", "STOP", "please hold", "waaaiiiiit"],
    "go_on": ["aiit", "alright", "go on", "move along", "proceed", "continue", "keep going", "carry on"],
    "note": ["so can you schedule this", "please set a timer", "please set this for that", "set this",
             "schedule this", "i have a schedule by this", "note this for that", "note this", "note", "take note",
             "this is important", "it's important"],
    "yes": ["ye", "sure", "yes", "okay", "ok", "yeah"],
    "no": ["noooo", "no", "NO", "noo"],
    "suggest": ["an example could be", "this is an example", "wait u can do it this way", "do it this way instead",
                "instead of this you could do it that way", "instead of this you could do that",
                "do that instead of this", "look at this", "you should look at this", "for example", "instead try",
                "try", "what if you do this", "what if", "what if we did this", "what of this", "what of",
                "anyways try", "maybe this instead", "maybe this"],
    "recall": ["i tried something yesterday can you remember", "can you remember", "can you recall",
               "can you remember this", "we did this", "i did this can you remember", "recall this and we will proceed",
               "there's this we did", "what's our most recent project", "what are the most recent projects",
               "our projects", "can you recall the project", "there's this project"],
    "strange": ["unusual", "this is strange", "weird isn't it", "it's strange", "it's weird",
                "it is surely something odd", "this feels off", "this is unusual", "this isn't normal",
                "whoah whoah something's wrong here", "what's going on", "can you understand what's happening",
                "what's happening here", "can you comprehend what's happening", "can you comprehend",
                "can you understand"],
    "repeat": ["please repeat that", "say it again", "please repeat", "repeat please", "say again", "huh", "err",
               "hmm", "repeat", "come again", "please can you say that again"],
    "wow": ["woaaah", "woah", "ooh myyy", "awesooome", "awesome", "woooooooooooo", "woooo", "whoa"],
    "who_said": ["who", "who said that", "who told you that", "who mentioned it", "do you know who said it",
                 "do you know what was said"],
    "where": ["where can we find it", "where is it", "where on earth would it be", "where is it now", "where",
              "where would it be"],
}

ADDED = {                                                       # more wordings around his, in the same voice
    "greet": ["yo coal", "wassup", "sup coal", "heyy", "yo yo", "what's good coal", "ayy"],
    "help": ["help me with this", "can you help", "i need help", "help me out here", "could you help me with this",
             "can you sort this out", "can you fix this for me", "i need you to look at this file",
             "please help me fix this", "can you do this for me", "i need a hand with this", "help please"],
    "can_you": ["can you do it", "you can do this right", "is this fixable", "can it be fixed",
                "do you think you can solve it", "is it doable", "can this be done"],
    "stop": ["hold on", "hold up", "one sec", "wait a minute", "stop stop", "hang on", "wait wait", "not yet",
             "stop that", "hold it"],
    "go_on": ["go ahead", "next", "keep it going", "go for it", "you can continue", "move on", "carry on then"],
    "note": ["remember this", "write this down", "save this for later", "set a reminder", "remind me later",
             "put this in the schedule", "don't forget this", "keep this in mind", "mark this"],
    "yes": ["yep", "yup", "of course", "correct", "that's right", "exactly", "right", "mhm", "true"],
    "no": ["nope", "nah", "not really", "that's wrong", "no no", "not that", "wrong"],
    "suggest": ["how about this", "try this instead", "do it like this", "what about this", "here's an idea",
                "try it another way", "maybe do it like this", "what if we try that", "like this for example"],
    "recall": ["do you remember", "remember what we did", "what did we do yesterday", "you remember this right",
               "what were we working on", "remind me what we did", "recall what i said"],
    "strange": ["that's odd", "something is off", "that's not right", "what is this", "this is weird",
                "that doesn't look normal", "what just happened"],
    "repeat": ["what did you say", "say that again", "pardon", "one more time", "i didn't hear you",
               "can you repeat that", "sorry what"],
    "wow": ["wow", "no way", "nice", "that's cool", "sick", "omg", "damn", "yooo nice"],
    "who_said": ["who said it", "who was that", "who said so", "who told you", "who was talking"],
    "where": ["where did it go", "where's it", "where can i find it", "where did we put it", "where is that",
              "where would i find it"],
}


def split():
    """(taught, held): {kind: [lines]}; every fifth of his own lines per kind is held."""
    taught, held = {}, {}
    for k, lines in MINE.items():
        held[k] = lines[2::5]
        taught[k] = [x for i, x in enumerate(lines) if i % 5 != 2] + ADDED.get(k, [])
    return taught, held
