# friend-supervisor
A long-term memory chatbot based on langchain, zep_cloud and streamlit. 

# Setup
1. Clone the repository
   ```
   git clone https://github.com/Tensionteng/friend-supervisor.git

   cd friend-supervisor
   ```
2. Install the required packages
   ```
   conda create -n friend-supervisor python=3.10

   conda activate friend-supervisor
   
   pip install -r requirements.txt
   ```

3. Set up the environment variables
   ```
    # create a .env file
    touch .env 

    # add the following lines to the .env file, replace "your_deepseek_api_key" with your actual deepseek api key
    DEEPSEEK_API_KEY="your_deepseek_api_key"

   ```
   or you can change any llm you want

4. Run the streamlit app
    ```
    streamlit run main.py
    ```