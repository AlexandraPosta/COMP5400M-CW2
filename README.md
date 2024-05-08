# COMP5400M-CW2

Use a sound source localisation algorithm as a sensory input for bio-inspired agents.

## Introduction

The purpose of this project is to explore the behaviour of bio-inspired agents when using a sound source localisation algorithm as a sensory input. The agents will be tested in a simulated environment with the aim of finding the best configuration for the agents to be able to locate the sound source.

## Sound Source Localisation Algorithm

The chosen agents are based on the cricket localisation algorithm. The algorithm is based on the time difference of arrival (TDOA) of the sound at the two ears of the cricket. The algorithm uses the TDOA to calculate the angle of the sound source relative to the cricket. The algorithm is implemented in the `sound_source_localisation.py` file.

## Project Structure

The project is structured as follows:

- The main library is in the folder `cricket_lib`. This folder is structured as a library and will be used by all other files.
- `cricket_mating` contains the code for analysing a single agent and its ability to locate the sound source.
- `cricket_race` contains the code for analysing multiple agents and their ability to locate the sound source.
- `ormia_response` contains the code for TODO
- All other folders contain resources such as the sound source or the saved models.

## Running the Project

### Libraries and other software

As this is a project that uses Python, it is recommended to use a virtual environment to run the project. The used version of Python is 3.10.2 and can also be found in the `.python-version` file if using `pyenv`.

We also recommend using a virtual environment manager such as `pyenv` or `virtualenv` to manage the virtual environment.

The required libraries can be found in the `requirements.txt` file. To install the required libraries, run the following command:

```bash
pip install -r requirements.txt
```

### Executing the script

The project is setup to be run using the `main.py` file.

As there are 3 different configurations, the script can be run with the following arguments:

- `mating`: Run the script for a single agent.
- `race`: Run the script for multiple agents.
- `ormia`: Run the script for the ormia response.

As an example, to run the script, use the following command:

```bash
python main.py mating
```

## Results

The results of the simulation will be displayed live in a matplotlib window. The results will show the agent's position and the sound source's position and how the agent is moving towards the sound source.
Furthermore, the results will also be saved inside the `output` folder as a `.png` file that will contain the last frame of the simulation.

## Conclusion

The project aims to analyse the agents capacity of locating the sound source. The results will be analysed and discussed in the final report attached to this project.
