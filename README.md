# MECA0062-1 Vibration Testing and Experimental Modal Analysis

## Experimental modal analysis of a plane structure

### Academic Year 2024 – 2025

#### Author: Victor Renkin s2306326

#### Rapport/Slides
The rapport of the project can be find in the link overleaf : [Rapport](https://www.overleaf.com/read/jgghjcfpsgkc#ee7e0b) --> Need to change the conclusion et introduction repating a lot. Note : 16.38/20

The slide of the project can be find in the link overlead : [Slide](https://www.overleaf.com/read/wrcmcpkkkdbx#db3e73). Note : 15/20

## Table of Contents
- [Introduction](#introduction)
- [Requirements](#requirements)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [AI](#ai)


## Introduction
This repository contains the code for the project of the course MECA0062-1 Vibration Testing and Experimental Modal Analysis. The project consists of the experimental modal analysis of a plane structure. The structure is excited by a hammer and the response is measured by accelerometers. The data is then processed to extract firs the eigenfrequencies with the CMIF method. Also the damping with the circle fit method, or the peak picking method.


## Reqirements
This project requires the packages listed in the `requirements.txt` file. To install these dependencies, ensure you have [Python](https://www.python.org/) and [pip](https://pip.pypa.io/en/stable/) installed, then run the following command:

```bash
pip install -r requirements.txt
```
## Usage

The script is divided into several tasks to reduce calculation time, as some processes, like the stabilization diagram, can take significant time. By default, all tasks will be executed, but you can choose to run only specific ones.

### Run all tasks (default)
To execute all tasks:

```bash
python src/main.py
```	
### Run only experimental modal analysis
If you want to run only the experimental modal analysis (graphs from the first lab, peak-picking method, and circle-fit method):
```bash
python src/main.py [-m] [--modal_analysis]
```

### Run only the stabilization diagram
To execute just the stabilization diagram:
```bash
python src/main.py [-s] [--stabilization_diagram]
```
### Run the modal analysis and comparison methods
To run the modal analysis along with the comparison method:
```bash	
python src/main.py [-d] [--detailed_analysis]
```
### Help comande line
For help on the command-line arguments, use the following command:

```bash
python src/main.py [-h] [--help]
```


## Project Structure

### **`data/`**
Contains the data from the first lab, second lab, modes extracted from SAMCEF using NX, the location of shocks, and the nodes used to represent the structure. Each category is organized into a separate folder:  
- `first_lab`: Data from the first laboratory.  
- `second_lab`: Data from the second laboratory.  
- `mode_samcef`: Modes extracted from SAMCEF.  

The nodes of the structure are represented in the files:  
- `node_structure.txt`  
- `node_shock.txt`  

---

### **`figures/`**
Contains the figures of the project, separated into:  
- **`first_lab`**: Includes figures for the CMIF, peak-picking method, and circle-fit method.  
- **`sec_lab`**: Contains figures related to the modes and their correlation with the structure.  

---

### **`src/`**
Contains all the code used in the project:  
- **`VizTool.py`**: Generates all graphical representations except those related to the modes.  
- **`DataPull.py`**: Handles data extraction.  
- **`main.py`**: Compiles and executes the entire project.  
- **`VizModes.py`**: Generates graphical representations of the modes and the structure.  
- **`SDOF_modal_analysis.py`**: Performs SDOF modal analysis, including CMIF, peak-picking method, and circle-fit method.  
- **`interpolate_data.py`**: Performs data interpolation. Although several functions are available, only linear interpolation is used.  
- **`polymax.py`**: Implements the Polymax method, stabilization diagrams, and LFSD.  
- **`comparaison_method.py`**: Compares methods using MAC and AutoMAC.  


## AI

The AI is used to occasionally correct the code and to reformulate sentences from the report.
