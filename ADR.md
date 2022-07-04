# Architecture Decision Records

We document key decisions of the project relevant to the 
data science implementation or software development with
lightweight records as part of the code base. This allows
contributors to follow the reasoning and compromises made
during the project while keeping documentation effort low.

For more details of the approach soo
[Github ADR](https://adr.github.io/)

As the first decision we decide to maintain the ADRs in this single file.

## Architecture Decision Record Template
This is a template for the records:

  In context of <feature f/use case/experiment>, 
  facing <concern c> (in <context x>)
  we decided for <option o> (and neglected <other options>), 
  to achieve <system qualities q/desired consequences>, 
  (accepting <downside d/undesired consequences>),
  because <additional rationale>.

Please add a few word title with a date. 
Add new entries at the end.

## 2022-06-20 Language and Framework

In the context of the initial project setup we decided on the language and frameworks.

### Language: Python 3.9+
We decided to use the latest version of Python because the language is
the obvious choice for a machine learning project and all initial 
participants are familiar with it (to different degrees).

We tried using the latest version of Python 3.10 but it doesn't work 
in a Cygwin environment under Windows. Thus we use 3.9 until 3.10 becomes
available for Cygwin.

Note that Python is also used for the site documentation (see below).

For the build process shell scripts and Makefiles may be used.

Other languages may be introduced later with judgment.

### Framework: Poetry
As a framework and dependency manager for Python we use 
[Poetry](https://python-poetry.org/docs/).

### Simulation-engine: Pettingzoo/Gym
Following recommendations from other brain-like AGI researchers 
we start development with the 
[Petting Zoo](https://github.com/Farama-Foundation/PettingZoo)
framework for simulating environments.

### Language Formatting
We format the Python source code with Black in the PEP8 style.
We tried japf with PEP8 but it caused differences with Black. 

## 2020-06-27 ADRs
In the context of the initial setup we discussed documentation requirements
and the need to keep the code-base understandable. We decided to use
lightweight architecture decision records that are stored
with a single-sentence format
in a single file 
as part of the code base.
This allows contributors to follow the reasoning and compromises made
during the project while keeping documentation effort low.
We accept that not all details are included and compensate by 
keeping the code readable and showing new project contributors how things work.
This means if you read this and are not sure what something means you should ask a the senior project member to show you around.

If the ADRs become unmanageable in a single file we may restructure later.

See also [Github ADR](https://adr.github.io/)


