## The Pluh team solution for Bigtarget hackathon.

This repository contains the code that was used for dataset analysis and the models optimizing uplift of test and control groups. The dataset was provided to us by the hackathon organizers at Lenta and Microsoft. The structure of the repository is the following:
1. [experiments](https://github.com/DKozl50/pluh_sms/tree/master/experiments) - The folder with jupyter notebooks used for dataset analysis and model tuning.
2. [features](https://github.com/DKozl50/pluh_sms/tree/master/features) contains functions for feature engineering.
3. [scripts](https://github.com/DKozl50/pluh_sms/tree/master/scripts) is the folder with some utility functions. Models are located in [pipeline](https://github.com/DKozl50/pluh_sms/blob/master/scripts/pipeline.py). The final model is BlendedModel that uses blending of several independent algorithms.
4. [submissions](https://github.com/DKozl50/pluh_sms/blob/master/submissions) - the name speaks for itself.
5. [site](https://github.com/DKozl50/pluh_sms/blob/master/site) - The source code of our dashboard.
