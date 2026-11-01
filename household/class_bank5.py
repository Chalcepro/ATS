"""Class lessons, batch 6 (10-31): the wider world, gaming, the alphabet and basic sums.
Chalce: a general personal model at a neutral world level, not one country's; he games a lot.
The alphabet and sums come first so Coal can later be handed a plain dictionary to read.
Same shape as class_bank.py; the fourth way of asking is held back.
"""
from household.class_bank2 import _what

WORLD = [
    ("chinese new year", "the spring festival in china, with red lanterns, dumplings, fireworks and family dinners"),
    ("a red envelope", "a red packet with money given at chinese new year for good luck"),
    ("a lantern", "a light in a paper or glass case; china holds a lantern festival"),
    ("the chinese zodiac", "twelve animals, one for each year, like the rat, dragon and tiger"),
    ("a panda", "a black and white bear from china that eats bamboo"),
    ("bamboo", "a tall fast-growing plant with a hollow stem, eaten by pandas"),
    ("silk", "a smooth shiny cloth first made in china from silkworm thread"),
    ("the silk road", "old trade routes that joined china to europe"),
    ("mandarin", "the most spoken language in china"),
    ("calligraphy", "the art of beautiful handwriting with a brush or pen"),
    ("anime", "japanese cartoons, often with big eyes and big stories"),
    ("manga", "japanese comic books, read from right to left"),
    ("christmas", "a festival on december 25th with gifts, trees and family"),
    ("halloween", "an october night when children dress up and collect sweets"),
    ("thanksgiving", "an american holiday in november to give thanks with a big dinner"),
    ("eid", "a muslim festival with prayers, feasts and gifts after ramadan"),
    ("diwali", "the indian festival of lights"),
    ("a birthday", "the day each year you were born, often with cake and presents"),
    ("the olympics", "the world's biggest sports games, held every four years"),
    ("the world cup", "the biggest football competition between countries, every four years"),
    ("football", "a game where two teams kick a ball to score in a goal; americans call it soccer"),
    ("basketball", "a game where two teams throw a ball through a high hoop"),
    ("chess", "a board game of thinking where you try to trap the other king"),
    ("the pyramids", "huge old stone tombs built in egypt thousands of years ago"),
    ("the eiffel tower", "a tall iron tower in paris, france"),
    ("the statue of liberty", "a giant statue of a woman holding a torch in new york"),
    ("the moon landing", "when people first walked on the moon in 1969"),
    ("a language", "the words and rules people use to talk, like english or mandarin"),
    ("english", "a language spoken all over the world, first from england"),
    ("a flag", "a cloth with colours and shapes that stands for a country"),
    ("a passport", "a small book that lets you travel to other countries"),
    ("a plane", "a machine with wings that flies people across the world"),
    ("a train", "a long line of carriages that runs on rails"),
    ("a ship", "a big boat that crosses the sea"),
    ("a rocket", "a machine that blasts off into space"),
    ("an astronaut", "a person who travels into space"),
    ("a robot", "a machine that can move and do jobs by itself"),
    ("artificial intelligence", "computer programs that can learn and think a little, like me"),
    ("a dinosaur", "a giant animal that lived millions of years ago and died out"),
    ("a volcano", "a mountain that can blow out hot melted rock called lava"),
    ("an earthquake", "when the ground shakes hard because rocks move deep down"),
    ("a rainbow", "a curve of colours in the sky when the sun shines through rain"),
]

GAMES = [
    ("a video game", "a game you play on a screen with a controller, keyboard or touch"),
    ("a controller", "the thing you hold in your hands to play games, with buttons and sticks"),
    ("a console", "a machine just for playing games on a tv, like a playstation or xbox"),
    ("a gaming pc", "a strong computer with a good graphics card for playing games"),
    ("a graphics card", "the part of the computer that draws the pictures, important for games"),
    ("fps in games", "frames per second: how smooth the game looks; more is smoother"),
    ("lag", "when a game is slow or jumpy because the computer or internet can't keep up"),
    ("ping", "how long the game takes to talk to the server; low ping is good"),
    ("a level", "one stage of a game, or how strong your character is"),
    ("a boss", "a big strong enemy, often at the end of a level"),
    ("a checkpoint", "a place in a game where you start again if you die"),
    ("a save point", "a place where the game keeps your progress"),
    ("a health bar", "the bar that shows how much life your character has left"),
    ("respawn", "coming back to life in a game after you die"),
    ("a quest", "a task or mission in a game"),
    ("an npc", "a non player character, someone in the game run by the computer"),
    ("a player", "a person playing the game"),
    ("multiplayer", "a game where many people play together"),
    ("single player", "a game you play by yourself"),
    ("co-op", "playing together on the same team"),
    ("pvp", "player versus player, when players fight each other"),
    ("a speedrun", "finishing a game as fast as possible"),
    ("a high score", "the best score anyone has got in a game"),
    ("an open world game", "a game with a big map you can explore freely"),
    ("a shooter", "a game where you aim and shoot"),
    ("a racing game", "a game where you drive fast to win races"),
    ("a puzzle game", "a game where you solve problems to move on"),
    ("a platformer", "a game where you run and jump between platforms"),
    ("an rpg", "a role playing game where your character grows stronger on a long story"),
    ("a game engine", "the program people use to build games"),
    ("a mod", "a change made by players to a game"),
    ("a glitch", "a mistake in a game that makes something weird happen"),
    ("a cheat code", "a secret code that gives you something special in a game"),
    ("an achievement", "a reward the game gives you for doing something special"),
    ("a skin", "a different look for a character or item in a game"),
    ("a loot drop", "items that fall when you beat an enemy"),
    ("a cooldown", "the wait time before you can use a move again"),
    ("rage quit", "leaving a game angrily because you lost"),
    ("gg", "good game, said at the end of a match to be friendly"),
    ("afk", "away from keyboard: the player is not there right now"),
    ("noob", "a new player who is still learning"),
    ("a tutorial", "the first part of a game that teaches you how to play"),
]

GAME_SAYS = [
    ("Let's play.", "okay! what are we playing?"),
    ("I won the match!", "gg! well played"),
    ("I lost again.", "you'll get it next time. want to try again?"),
    ("This boss is too hard.", "watch its moves, find the pattern, and try again"),
    ("The game is lagging.", "close other programs and check the internet"),
    ("My controller died.", "plug it in to charge, or use the keyboard for now"),
    ("I'm stuck on this level.", "let's look for another way, or check a guide"),
    ("I got a new game.", "nice! what kind of game is it?"),
    ("I beat the game!", "wow, well done! that's a big win"),
    ("Gg.", "gg, well played"),
]

ALPHABET = [(c, f"the letter {c}, number {i + 1} in the alphabet") for i, c in enumerate("abcdefghijklmnopqrstuvwxyz")]
LETTER_LESSONS = ([("English", [f"What letter comes after {a}?", f"Which letter is after {a}?", f"After {a} comes which letter?",
                                 f"Say the letter that follows {a}."], b) for a, b in zip("abcdefghijklmnopqrstuvwxy", "bcdefghijklmnopqrstuvwxyz")]
                  + [("English", [f"What letter comes before {b}?", f"Which letter is before {b}?", f"Before {b} comes which letter?",
                                   f"Say the letter just before {b}."], a) for a, b in zip("abcdefghijklmnopqrstuvwxy", "bcdefghijklmnopqrstuvwxyz")]
                  + [("English", ["Say the alphabet.", "What are the letters of the alphabet?", "Sing the abc.",
                                   "Tell me all the letters from a to z."], " ".join("abcdefghijklmnopqrstuvwxyz"))]
                  + [("English", ["What are the vowels?", "Which letters are vowels?", "Name the vowels.",
                                   "Tell me the vowel letters."], "a, e, i, o and u")])

N = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
     "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty"]
SUMS = ([("maths", [f"What is {a} plus {b}?", f"What is {a} + {b}?", f"Add {a} and {b}.", f"If you have {a} and get {b} more, how many?"],
          f"{a + b}, {N[a + b]}") for a in range(1, 11) for b in range(1, 11) if a <= b]
        + [("maths", [f"What is {a} minus {b}?", f"What is {a} - {b}?", f"Take {b} away from {a}.", f"If you have {a} and give away {b}, how many are left?"],
            f"{a - b}, {N[a - b]}") for a in range(2, 11) for b in range(1, a)]
        + [("maths", [f"How do you write {N[n]} as a number?", f"What number is {N[n]}?", f"Write {N[n]} in digits.",
                      f"{N[n]} is which number?"], str(n)) for n in range(21)]
        + [("maths", [f"What comes after {n}?", f"Which number is after {n}?", f"Count on from {n}: what is next?",
                      f"What number follows {n}?"], f"{n + 1}") for n in range(20)])

MORE5 = ([_what(t, a, "world") for t, a in WORLD] + [_what(t, a, "games") for t, a in GAMES]
         + [("games", [f"Someone says: {s} What do you say back?", f"A person tells you '{s}' How do you reply?",
                       f"What is a good answer to '{s}'", f"If they say {s.lower().rstrip('.!?')}, what would you say?"], a)
            for s, a in GAME_SAYS]
         + [_what(f"the letter {c}", a, "English") for c, a in ALPHABET] + LETTER_LESSONS + SUMS)
