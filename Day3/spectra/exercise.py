Answer = [
    [8, 64, 511],
     [4, 128,169],
     [2,16,25],
     [3., 5., 4., 8., 6.],
     [-1.8000, -1.8000, -1.8000],
     [2.0000, 3.5000, 2.0000, 7.5000, 3.5000]
]


def check_question(Exercise,Correct_Answer):
    if Exercise == Correct_Answer:
        print("You are right!")
    else:
        print("You are wrong!")



def check_question_one(Exercise):
    check_question(Exercise,Answer[0])


def check_question_second(Exercise):
    check_question(Exercise,Answer[1])
    
def check_question_third(Exercise):
    check_question(Exercise, Answer[2])
    
def check_question_fourth(Exercise):
    check_question(Exercise, Answer[3])
    
def check_question_fifth(Exercise):
    check_question(Exercise, Answer[4])
    
def check_question_sixth(Exercise):
    check_question(Exercise, Answer[5])
