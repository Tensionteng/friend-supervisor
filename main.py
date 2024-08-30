import utils
import streamlit as st
from streaming import StreamHandler

from dotenv import load_dotenv
import os
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from prompt_template import system_prompt

st.set_page_config(page_title="Context aware chatbot", page_icon="⭐")
st.header("Context aware chatbot")
st.write("Enhancing Chatbot Interactions through Context Awareness")
st.write(
    "[![view source code ](https://img.shields.io/badge/view_source_code-gray?logo=github)](https://github.com/shashankdeshpande/langchain-chatbot/blob/master/pages/2_%E2%AD%90_context_aware_chatbot.py)"
)


class ContextChatbot:

    def __init__(self, api_key: str = os.getenv("DEEPSEEK_API_KEY")):
        utils.sync_st_session()
        self.llm = utils.configure_llm(api_key=api_key)

    @st.cache_resource(ttl=60 * 60 * 1)
    def setup_chain(_self):
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    system_prompt,
                ),
                MessagesPlaceholder(variable_name="history"),
                # MessagesPlaceholder(variable_name="sumary"),
                ("human", "{input}"),
            ]
        )
        # memory = ConversationBufferMemory()
        # chain = ConversationChain(llm=_self.llm, memory=memory, verbose=False)
        return prompt | _self.llm

    @utils.enable_chat_history
    def main(self):
        chain = self.setup_chain()
        user_query = st.chat_input(placeholder="Ask me anything!")
        if user_query:
            utils.display_msg(user_query, "user")
            with st.chat_message("assistant"):
                st_cb = StreamHandler(st.empty())

                result = chain.invoke(
                    {"input": user_query, "history": st.session_state.messages},
                    {"callbacks": [st_cb]},
                )
                # result = chain.invoke({"input": user_query}, {"callbacks": [st_cb]})

                response = result.content
                # st.write_stream(response)
                st.session_state.messages.append(
                    {"role": "assistant", "content": response}
                )
                utils.print_qa(ContextChatbot, user_query, response)


if __name__ == "__main__":
    load_dotenv()
    obj = ContextChatbot()
    obj.main()
