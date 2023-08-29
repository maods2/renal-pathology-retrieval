# Renal Pathology retrieval

This repository, `renal-pathology-retrieval`, contains the backend of the Terumo application.

## Terumo: Contact-Based Image Retrieval Tool for Renal Pathologists

Terumo is a web software that serves as a contact-based image retrieval tool for renal pathologists. It allows users to perform queries using images of renal biopsies of glomeruli and retrieves images with semantic similarities.

At the moment, this API is a monolith that was initially created with the intention of becoming a microservice app. However, it currently functions as a monolith and provides the following features:

- Serving the images saved in the database
- Managing metadata of the collection of images saved in the database
- Performing content-based image retrieval, including similarity calculations and the implementation of machine learning models

This API implements 6 EfficientNet-B0 binary models, with each model was trained to detect a specific type of renal lesion and extract semantic attributes used for calculating similarity among the images in the database. Additional models for extracting embeddings to aid in the search for similar images will be implemented in the future. The API currently supports the following renal lesion models:

- Normal glomeruli
- Hypercellularity
- Sclerosis
- Membranous
- Podocytopathy
- Crescent


```


```


## Running the Application

To run the application, follow these steps:

1. Install Python 3.10.4.
2. Install pipenv, a package manager for Python:
   - Install pipenv using pip: `pip install pipenv`
   - Create a project and set the Python version: `pipenv --python 3.10.4`
   - Install the packages listed in an existing Pipfile.lock: `pipenv sync`
   3. Run the application.
      1. Download Image Dataset and Extract the Zip File
   >   Download the image dataset required by the Terumo application. Once downloaded, extract the contents of the zip file to a directory of your choice ( The image dataset used in the indexing database is located on the PattoSpother share drive).
      
      3. Launch pipenv environment and start the app
      ```bash
      cd /path/to/your/project
      pipenv shell
      python src/app.py
      ```
3. Development:
   - Automating tests with task py.
   >  In order to automate process we are using taskpy. You can check the tasks available using the  command. 
    ```bash
    task -l
    ```
   - Cleaning up the repository with Make.
   ```bash
   make clean
   ```
   - Installing libraries for different environments.
   ```bash
   pipenv install requests
   
   # Dev dependencies
   pipenv install --dev pytest
   pipenv install --dev pytest-cov 
   pipenv install --dev blue # PEP8 Formater
   pipenv install --dev isort # import sorting
   pipenv install --dev taskipy # automation scripting
   pipenv install --dev safety
   
   pipenv install --dev safety pytest pytest-cov blue isort taskipy
   ```
   - Saving libraries in requirements.
   ```bash
   pipenv requirements > requirements.txt
   pipenv requirements --dev-only > requirements.txt
   pipenv requirements --dev > dev-requirements.txt
   ```
   - Running unit tests
   ```bash
   task test 
   ```  
   - Generating code coverage report
   ```bash
   task post_test 
   ```
   - Running linter
   ```bash
   task lintr 
   ```
   - Checking libs vulnerabilities
   ```bash
   pipenv check
   ```   

4. Packing the application as a container:
   - Building the container.
   ```bash
   sh scripts/build_container.sh
   ```
   - Running the container.
   ```bash
   sh scripts/run_container.sh
   ```
   - Pushing to a remote registry.
   ```bash
   sh scripts/push_image.sh
   ```
   
Please refer to the specific sections in this repository for detailed instructions on each step.

## Conclusion

You are now familiar with the Terumo Service Search Monolith repository. This backend API, implemented in Python, provides functionalities for serving images, managing metadata, and performing content-based image retrieval for renal pathologists. Follow the provided instructions to set up, run, and develop the application according to your requirements.