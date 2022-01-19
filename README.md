# STDNN-Platform

*An end-to-end, experiment-oriented machine learning pipeline for spatial-temporal data*

## Project Information

### Current Development Team

| Member          | 
| -------------   |
| Carl Combrinck  |
| Jane Imrie      |

### Supervisor

| Staff Member              | Department           |
| -------------             |-------------         |
| A/Prof Deshendran Moodley | Computer Science     |

## Installation

```
make install
```
or
```
python3 -m venv venv
source ./venv/bin/activate
pip3 install -Ur requirements.txt
```

## Testing GWN Model
```
make test
```
or 
```
source ./venv/bin/activate
python3 user_main.py --model GWN --window_size 40 --horizon 10 --baseline True
```