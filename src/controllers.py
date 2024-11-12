import xlam_cpp  # Salesforce/xLAM-1b-fc-r API wrapper for LLM queries

# External Registries (Global)
tools_registry = {
    # Example format
    "WeatherAPI": ["Weather", "Provides weather forecasts and current weather data", "weather_api_call()"],
    # Add other tools similarly...
}

sub_controllers_registry = {
    # Example format
    "Weather": "Handles weather-related tasks using WeatherAPI to provide weather information.",
    # Add other sub-controller domains similarly...
}

metadata_registry = {
    # Metadata for tools, sub-controllers, and domains.
    "WeatherAPI": {"domain": "Weather", "capabilities": ["forecast", "current weather"], "version": "1.0"},
    # Add metadata for other tools and SCs...
}

# GeneralController Class
class GeneralController:
    def __init__(self, sub_controllers, gsc, llm_api_key, max_feedback_loops=3):
        self.sub_controllers = sub_controllers  # Dictionary of Sub-Controller instances
        self.gsc = gsc  # General Sub-Controller for handling common tasks
        self.llm_api_key = llm_api_key
        self.max_feedback_loops = max_feedback_loops
        self.plugins = {}  # Storage for dynamically registered plug-ins

        # Copy of permanent registries for temporary use during task lifecycle
        self.task_sub_controllers = sub_controllers_registry.copy()
        self.task_tools = tools_registry.copy()

    def process_query(self, query):
        # Decompose the query into subtasks with dynamic role assignment
        tasks = self.decompose_query(query)
        responses = {}

        # Execute each subtask and provide feedback if necessary
        for role, task in tasks.items():
            if role in self.sub_controllers:
                responses[role] = self.execute_with_feedback(self.sub_controllers[role], task)
            else:
                # Attempt to handle with the General Sub-Controller (GSC) first
                gsc_response = self.execute_with_feedback(self.gsc, task)
                if not self.verify_result(gsc_response):
                    # If GSC cannot handle the task effectively, create a new SC
                    responses[role] = self.create_new_sc(role, task)
                else:
                    responses[role] = gsc_response

        # Aggregate responses from all subtasks and return the final result
        final_response = self.aggregate_responses(responses)
        self.store_persistent_data(query, final_response, responses)
        return final_response

    def register_plugin(self, plugin_name, plugin_instance):
        """Registers a new external plug-in sub-controller"""
        self.plugins[plugin_name] = plugin_instance
        print(f"Plug-in {plugin_name} has been registered successfully.")

    def decompose_query(self, query):
        # Use LLM to break down the query into subtasks and assign appropriate roles
        prompt = (
            f"Decompose the following query into subtasks and assign roles.\nQuery: {query}\n"
            f"Available domains and capabilities: {self.task_sub_controllers}\n"
            "For each subtask, provide a score (0-10) to rank which domain is best suited. "
            "If no domain seems appropriate, create a new domain."
        )
        response = self.llm_query(prompt)

        tasks = {}
        for line in response.splitlines():
            role, score, task = line.split(': ', 2)
            score = int(score)
            if score >= 8:  # Threshold for matching an SC
                tasks[role] = task
            else:
                tasks["general"] = task  # Assign to GSC if no match is found

        return tasks

    def execute_with_feedback(self, sc, task):
        feedback_list = []  # List to hold feedback history
        for _ in range(self.max_feedback_loops):
            result = sc.handle_task(task, feedback=feedback_list)
            if self.verify_result(result):
                return result
            # Generate feedback and append to the feedback list
            feedback = self.generate_feedback(task, result)
            feedback_list.append(feedback)

            if "converged" in feedback.lower():
                break

        return result

    def verify_result(self, result):
        # Verify quality of result using LLM
        prompt = f"Verify if this result meets the requirements: {result}\nProvide a 'pass' or 'fail' response."
        verification_response = self.llm_query(prompt)
        return "pass" in verification_response.lower()

    def generate_feedback(self, task, result):
        # Generate feedback for improving the result
        prompt = f"Provide detailed feedback on how to improve the following result.\nTask: {task}\nResult: {result}"
        return self.llm_query(prompt)

    def create_new_sc(self, role, task):
        # Create a new sub-controller dynamically for a new role
        prompt = (
            f"Create a new sub-controller for domain '{role}' to handle this task.\n"
            f"Take reference from existing sub-controllers to provide comprehensive instructions."
        )
        sc_instructions = self.llm_query(prompt)
        new_sc = DynamicSubController(role, sc_instructions)
        self.task_sub_controllers[role] = f"Handles domain '{role}' tasks as per given instructions."
        return new_sc.handle_task(task)

    def store_persistent_data(self, query, final_response, responses):
        # Store successful SCs or tools into permanent registries
        for role, response in responses.items():
            if "success" in response.lower():
                if role not in sub_controllers_registry:
                    sub_controllers_registry[role] = self.task_sub_controllers[role]

    def aggregate_responses(self, responses):
        # Aggregate results meaningfully
        prompt = (
            "Combine the following subtasks' results into a meaningful response to answer the original query.\n"
            f"Subtask results: {responses}\nProvide a combined response."
        )
        return self.llm_query(prompt)

    def llm_query(self, prompt):
        # Make LLM API call using Salesforce/xLAM-1b-fc-r
        response = xlam_cpp.Completion.create(
            model="xlam-1b-fc-r",  # Adjust model if needed
            prompt=prompt,
            max_tokens=150,
            temperature=0.7
        )
        return response.choices[0].text.strip()

# SubController Class
class SubController:
    def __init__(self, role, tools, capabilities):
        self.role = role
        self.tools = tools
        self.capabilities = capabilities

    def handle_task(self, task, feedback=[]):
        prompt = (
            f"As a domain expert in '{self.role}', you have the following tools: {self.tools}.\n"
            f"Capabilities: {self.capabilities}\nTask: {task}\n"
            f"Feedback: {' | '.join(feedback)}\nProvide an expert response."
        )
        return xlam_cpp.Completion.create(
            model="xlam-1b-fc-r",
            prompt=prompt,
            max_tokens=200,
            temperature=0.7
        ).choices[0].text.strip()

# GeneralSubController Class
class GeneralSubController(SubController):
    def __init__(self, role="general", tools=None):
        capabilities = "Handles generic tasks that do not fall under any specialized domain."
        super().__init__(role, tools, capabilities)

    def handle_task(self, task, feedback=[]):
        prompt = (
            f"As the general-purpose controller, you are handling the following task: {task}.\n"
            f"Feedback history: {' | '.join(feedback)}\nProvide a response."
        )
        return xlam_cpp.Completion.create(
            model="xlam-1b-fc-r",
            prompt=prompt,
            max_tokens=200,
            temperature=0.7
        ).choices[0].text.strip()
