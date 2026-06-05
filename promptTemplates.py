from langchain_core.prompts import ChatPromptTemplate

prompt_message=ChatPromptTemplate([("system":"you are translation agent who gives perfect trnaslation to the user"),
                                  ("user":"translate this sentence{input} into {target_language}")])

prompt_message.invoke({"input":"i go to school","target_language":"french"})
