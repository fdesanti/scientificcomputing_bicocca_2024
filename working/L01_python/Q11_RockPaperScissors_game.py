"""
Code implementation for the exercise Q11 - Rock Paper Scissors Game
"""
import random

class RockPaperScissorGame:
    """
        Remember the rules:
         - Rock beats scissors
         - Scissors beat paper
         - Paper beats rock
    """
    def __init__(self, mode='classic'):

        assert mode in ['classic', 'TBBT'], "Mode must be either 'classic' or 'TBBT' (The Big Bang Theory version)"
        #wins counter
        self._wins = {"User"    : 0, 
                      "Computer": 0}
        #plays counter
        self._plays   = 0

        #possible choices
        if mode == 'classic':
            self._choices = ["rock", "paper", "scissors"]
            self._approx={'r': 'rock',
                          'p': 'paper',
                          's': 'scissors'}
        
        elif mode == 'TBBT':
            self._choices = ["rock", "paper", "scissors", "lizard", "spock"]
            self._approx={'r' : 'rock',
                          'p' : 'paper',
                          's' : 'scissors',
                          'l' : 'lizard',
                          'sp': 'spock'}
        
        #dictionary saying what each choice beats    
        self._beats = {"rock"    : ["scissors", "lizard"], 
                       "scissors": ["paper", "lizard"],
                       "paper"   : ["rock", "spock"],
                       "lizard"  : ["spock", "paper"],
                       "spock"   : ["scissors", "rock"]}
        

    @property
    def wins(self):
        return self._wins
    
    @property
    def total_plays(self):
        return self._plays
    
    @property
    def available_choices(self):
        """
            Returns a string with the available choices for the user to choose from
            Example: "rock, paper, scissors" or "rock, paper, scissors, lizard, spock"
        """
        return [c+'/'+self._approx[c] for c in self._approx.keys()]
    
    def _format_choice(self, choice):
        """
            Format the user's input choice to be a valid choice. 
            Example: if the user types 'r', it will be converted to 'rock'
        """
        if choice in self._approx.keys():
            return self._approx[choice]
        elif choice in self._choices:
            return choice
        else:
            raise ValueError(f"invalid choice: available are {self.available_choices}")
    
    def _who_wins(self, user_choice, computer_choice):
        """ 
            Determine who wins from the user's perspective. 
            First we check if both chose the same options, 
            then we check if the user's choice beats the computer's choice.

            Args:
            -----
                user_choice (str)    : the user's choice
                computer_choice (str): the computer's choice
            
            Returns:
            --------
                str: the winner
        """
        #determine what the user's choice beats
        user_beats = self._beats[user_choice]
        print(f"User choice: {user_choice}, Computer choice: {computer_choice}")
        if user_choice == computer_choice:
            print("Tie")
            return "Tie"
        if any([computer_choice == c for c in user_beats]):
            print("Congratulations, you win! ;)")
            return "User"
        else:
            print("Sorry, computer wins.. ")
            return "Computer"

    def _get_user_choice(self):
        """ get the user's choice from command line """
        #choice = input(f"Please choose among {[s[0]+'/'+str(s) for s in self._choices]}")  
        msg = f"\nPlease choose among {self.available_choices}:"
        choice = input(msg).lower()
        return self._format_choice(choice)
    
    def _get_computer_choice(self):
        """ get the computer's choice by randomly picking one of the choices """
        return random.choice(self._choices)
    
    def _update_win_counter(self, winner):
        """ add a win to the appropriate counter and update plays counter"""
        if not winner == "Tie":
            self._wins[winner] += 1

        #anyway we update the play counter
        self._plays += 1

    def print_outcome(self):
        """ prints the outcome of the game """
        print("\nGame over!")
        print("----------------")
        print(f"Total plays: {self.total_plays}")
        print(f"User won: {self._wins['User']}\nComputer won: {self._wins['Computer']}")
        print("----------------")

    def play(self):
        """ Main function to play the game """

        while True:
            try:
                user_choice = self._get_user_choice()
            except Exception as e:
                print(e)
                print("Please try again...")
                continue
            computer_choice = self._get_computer_choice()
            winner = self._who_wins(user_choice, computer_choice)
            self._update_win_counter(winner)

            go_on = input("Do you want to play again? (y/n)")
            if go_on == "n":
                break
        
        self.print_outcome()
        return self._wins
        

if __name__ == "__main__":
    game = RockPaperScissorGame(mode='TBBT')
    plays = game.play()

    
