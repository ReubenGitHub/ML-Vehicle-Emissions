# <b>Machine Learning: An Investigation into Vehicle CO2 Emissions</b>

In this project, I investigate the recorded CO2 emissions (WLTP) of 6756 vehicles, and use supervised machine learning techniques to predict CO2 emissions from vehicle features, including the type of powertrain and size of the engine.

The data was obtained via the following UK government link:  
https://carfueldata.vehicle-certification-agency.gov.uk/downloads/default.aspx  
Note that the data at this link is subject to change. The raw data as I obtained it can be found in the data/raw directory [here](data/raw/Euro_6_latest.csv).

In the [Project Report file](ProjectReport.md), I offer a report of the work performed in this project, describing the fundamental stages and key results.

[Trieste citation](trieste_citation.txt)

## License
[Apache 2.0 License](LICENSE)

## Getting Started

It is not required to execute any of the code in this repository in order to obtain and review the results of this investigation. The stages and key results are described in the ProjectReport.md file, with some other secondary results being shown in the various "..._vis" folders of the project.

If you wish to follow along in a practical sense, for example by reproducing visualisations or training models, or if you wish to run any of the code for any other reason, then the Python dependencies are listed in the [requirements.txt](requirements.txt) file.

In addition to these libraries, you will also require an installation of MySQL. You might find the following tutorials helpful in getting set up:  
Windows: https://www.youtube.com/watch?v=2HQC94la6go&ab_channel=BlaineRobertson  
Mac: https://www.youtube.com/watch?v=5BQ5GvjiAR4&ab_channel=BlaineRobertson  
Linux: https://www.youtube.com/watch?v=0o0tSaVQfV4&ab_channel=webpwnized

As you install MySQL, make sure your username and password match those stored in [configprivate.py](src/data/dbLogin/configprivate.py). In the development of this project I used username = "root" and password = "password123".

If you wish to follow along, the Python scripts should be executed in the following order:
1. [dataInitialise.py](src/data/dataInitialise.py)
2. [dataInitialiseAnalyse.py](src/data/dataInitialiseAnalyse.py)
3. [dataClean.py](src/data/dataClean.py)
4. [dataCleanAnalyse.py](src/data/dataCleanAnalyse.py)
5. [dataPreprocess.py](src/data/dataPreprocess.py)
6. [dataPreprocessAnalyse.py](src/data/dataPreprocessAnalyse.py)
7. [trainTestAnalyse.py](src/models/trainTestAnalyse.py)
8. [train.py](src/models/train.py)  
9. [trainTunedOrdinal.py](src/models/trainTunedOrdinal.py)  
10. [trainTunedOnehot.py](src/models/trainTunedOnehot.py)  
11. [trainTunedOrdinalNoMan.py](src/models/trainTunedOrdinalNoMan.py)
12. [modelReview.py](src/models/modelReview.py)

Many of these files print text-based reports of their function and results, and tuning scripts take a while to run if fully executed, which is why I have not created a main script to run these files sequentially.