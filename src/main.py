from controllers import GeneralController, SubController, GeneralSubController, DynamicSubController
import openai
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# Retrieve API key from environment variable for better security
llm_api_key = os.getenv("OPENAI_API_KEY")  # We need to replace this with our actual key

# Create a general-purpose sub-controller
class AnalysisSubController(SubController):
    def handle_task(self, task):
        # Override handle_task for domain-specific analysis
        prompt = f"Performing detailed analysis on the following task: {task}"
        return self.llm_query(prompt)

    def llm_query(self, prompt):
        # LLM API call with error handling
        try:
            response = openai.Completion.create(
                model="gpt-4",
                prompt=prompt,
                max_tokens=150,
                temperature=0.7
            )
            return response.choices[0].text.strip()
        except openai.error.OpenAIError as e:
            logging.error(f"Error in LLM query for AnalysisSubController: {e}")
            return "Error handling the task"

# Instantiate sub-controllers
analysis_sc = AnalysisSubController("analysis", llm_api_key)
gsc = GeneralSubController(llm_api_key)  # General sub-controller for fallback tasks

# Initialize the General Controller with sub-controllers and the GSC
sub_controllers = {"analysis": analysis_sc}
gc = GeneralController(sub_controllers, gsc, llm_api_key)

# Test a sample query using the General Controller
query = "Analyze customer sales data trends"
logging.info(f"Processing query: {query}")
response = gc.process_query(query)
print("Final Response:", response)

# Test a query that requires dynamic sub-controller creation
query_dynamic = "Handle a complex new financial task"
logging.info(f"\nProcessing new dynamic query: {query_dynamic}")
response_dynamic = gc.process_query(query_dynamic)
print("Final Response (Dynamic):", response_dynamic)
