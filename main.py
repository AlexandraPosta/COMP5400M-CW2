"""
	COMP5400M - CW2
    Author Name: Alexandra Posta - el19a2p
                 Alexandre Monk - el19a2m
                 Bogdan-Alexandru Ciurea - sc20bac
"""

import sys
from cricket_race.main import main as cricket_race_main
from cricket_mating.main import main as cricket_mating_main
from ormia_response.main import main as ormia_audio_response_main

if __name__ == "__main__":
    # Take the arguments from the command line
    if len(sys.argv) != 2:
        print("Please provide the name of the experiment to run")
        sys.exit(0)

    experiment_name = sys.argv[1]

    # Run the experiment
    match experiment_name:
        case "mating":
            cricket_mating_main()
        case "race":
            cricket_race_main()
        case "ormia":
            ormia_audio_response_main()
        case _:
            print("Invalid experiment name")
            sys.exit(0)
