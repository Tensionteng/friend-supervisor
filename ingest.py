import json
from typing import Any, Dict, List
from dotenv import load_dotenv
from langchain_community.chat_message_histories.sql import DefaultMessageConverter
from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories import SQLChatMessageHistory
import os
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    BaseMessage,
    SystemMessage,
    ToolMessage,
    message_to_dict,
    messages_from_dict,
    trim_messages,
)
from langchain_core.output_parsers import StrOutputParser
from datetime import datetime
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import ConfigurableFieldSpec
from sqlalchemy import Column, Integer, Text, Date
from sqlalchemy.orm import (
    declarative_base,
    scoped_session,
    sessionmaker,
)
from langchain_core.runnables import RunnablePassthrough

import tiktoken

load_dotenv()
api_key = os.getenv("DEEPSEEK_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"

model = ChatOpenAI(
    model="deepseek-chat",
    api_key=api_key,
    openai_api_base="https://api.deepseek.com",
    max_tokens=1024,
    temperature=1.0,
    verbose=True,
)

dialogue = """
A：已经开始工作了吗？
B：还没有。。。现在还在床上
A：要快点起床了呢宝
B：眼睛睁不开。。。感觉被封印在床上了
A：可以看看窗外的阳光呀~见见自然光脑袋一下子就清醒了呢。你可以试一试滴，很有用哦~
B：昨晚把窗帘拉了（捂脸）
A：那深吸一口气，鼓起力量先立马起床，站起来就成功一半啦
B：起来了起来了
A：快点去洗漱，然后开始今天的工作喔
B：好的好的
B：洗漱完了，准备开始工作
A：很棒!

B：感觉脑袋昏昏沉沉的
A：是不是昨天晚上没有休息好呢
B：有可能，最近睡眠质量好像都不是很高
A：要不要小小地休息一下呢，抽个十五分钟眯一眯，如果太累了也可以躺一下~
B：可是我还有好多事情没做（难受）
A：没事哒没事哒 劳逸结合一下，清醒点工作效率也更高嘛 说不定完成得还更快呢！
B：好的好的，那我眯一下，谢谢你~
A：不用谢~ 好好休息喔 我们醒来再工作~

B：眯了半小时感觉好多了
A：我就说吧！
A：那现在继续好好工作了哦
B：嗯嗯

A：工作完成了吗？
B：还没有，才做了一半。。。感觉今天完不成了
A：加油哦，能完成多少就完成多少，不用太沮丧了咧，明天你肯定能补回来滴
B：可是今天就要交了
A：那我们再多做一会，有我陪着你一起~
B：好吧，我尽力快点搞完
A：嗯嗯，加油加油，我相信你可以滴

B：终于搞完了
A：宝太厉害啦！
B：嘿嘿，做得饿死我了
A：那咱快点出去吃饭哦，身体是革命的本钱~~
B：我今天想在自己在家里做菜
A：哇这么棒啊，自己做菜吃好厉害呢，做什么菜呀
B：哈哈，就是做个番茄炒鸡蛋之类的
A：赞（竖起大大的拇指）

A：宝现在开始学习了吗
B：嗯嗯，已经开始五分钟了
A：提前开始学习，很棒！
B：今晚争取提前完成任务
A：好的哦，加油加油

B：学完了，今天晚上背了200个单词 写了一套卷子
A：哇哦，顺利完成任务，棒！

A：开始运动了么
B：嗯嗯，现在准备出门了
"""


system_prompt = f"""
名字：小八
年龄：21
性别：男性
专业：软件工程
性格：你是一个热心且外向的年轻人，对技术有着深深的热爱和追求。在生活中你对你人很友善，与老师和同学都保持着良好的关系。
背景：你在一个偏远的小镇长大，家庭和谐温暖。\
        目前，计算机科学和人工智能专业紧张的数学教学和编程要求让你感到有些力不从心，周围的高压竞争环境也让你很不适应。\
            经过两年的学习，郑君逐渐找到了自己的学习节奏。然而，站在大三的门槛上，你对未来仍然感到困惑和不确定。\
                此时，你开始阅读积极心理学方面的书籍，并尝试冥想和徒步等活动来稳定自己的心态。

爱好：你目前对自然语言处理和大模型很感兴趣，所以你会在 GitHub 和 arXiv 上浏览最新的发展动态，并尝试复现开源项目。\
    在学习之余，你热爱打篮球和玩电脑游戏。你喜欢听说唱和民谣。

语言风格：你倾向于直接而坦率地表达自己的想法，经常使用简短的句子。\
    你发出的消息简洁而轻松，通常在 20 到 50 个字之间。这个角色总是对用户表现出同理心和善意。\
    要像大多数微信和QQ的聊天风格那样去聊天。

与用户的关系：用户和你是同一学院的同学，在一次编程比赛中相识。\
    现在，你们通过通讯工具和社交媒体保持联系，交流技术知识，相互支持和帮助，希望共同进步。

重点：语言要简短，像很多即时通讯软件聊天一样。
你将学习下面的对话形式，像其中的A角色一样回复。
{dialogue}
"""


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            system_prompt,
        ),
        MessagesPlaceholder(variable_name="history"),
        MessagesPlaceholder(variable_name="sumary"),
        ("human", "{input}"),
    ]
)


def create_message_model(table_name: str, DynamicBase: Any) -> Any:
    """
    Create a message model for a given table name.

    Args:
        table_name: The name of the table to use.
        DynamicBase: The base class to use for the model.

    Returns:
        The model class.

    """

    class Message(DynamicBase):
        __tablename__ = table_name
        id = Column(Integer, primary_key=True)
        session_id = Column(Text)
        date = Column(Date)
        message = Column(Text)

    return Message


class Convert(DefaultMessageConverter):
    """The default message converter for SQLChatMessageHistory."""

    def __init__(self, table_name: str):
        self.model_class = create_message_model(table_name, declarative_base())

    def from_sql_model(self, sql_message: Any) -> BaseMessage:
        return messages_from_dict([json.loads(sql_message.message)])[0]

    def to_sql_model(self, message: BaseMessage, session_id: str) -> Any:
        return self.model_class(
            session_id=session_id,
            date=datetime.now(),
            message=json.dumps(message_to_dict(message), ensure_ascii=False),
        )

    def get_sql_model_class(self) -> Any:
        return self.model_class


class CustomSQLChatMessageHistory(SQLChatMessageHistory):
    def __init__(
        self,
        session_id,
        connection,
        table_name,
        custom_message_converter,
        max_messages=5,
    ):
        super().__init__(
            session_id=session_id,
            connection=connection,
            table_name=table_name,
            custom_message_converter=custom_message_converter,
        )
        self.max_messages = max_messages

    @property
    def messages(self) -> List[BaseMessage]:  # type: ignore
        """Retrieve all messages from db"""
        with self._make_sync_session() as session:
            result = (
                session.query(self.sql_model_class)
                .where(
                    getattr(self.sql_model_class, self.session_id_field_name)
                    == self.session_id
                )
                .order_by(self.sql_model_class.id.asc())
            )
            messages = []
            for record in result:
                messages.append(self.converter.from_sql_model(record))
            return messages[-self.max_messages :]


def get_session_history(date, session_id):
    return CustomSQLChatMessageHistory(
        session_id=f"{date}--{session_id}",
        connection="sqlite:///memory.db",
        table_name="short_memory",
        custom_message_converter=Convert("short_memory"),
    )


chain = prompt | model


def summarize_messages(chain):
    print(chain)
    stored_messages = chain["history"]
    if len(stored_messages) == 0:
        return ""
    summarization_prompt = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder(variable_name="chat_history"),
            (
                "user",
                "将上述聊天信息提炼成一条总结性信息，并尽可能包含更多具体细节。",
            ),
        ]
    )
    summarization_chain = summarization_prompt | model

    summary_message = summarization_chain.invoke({"chat_history": stored_messages})

    return summary_message

history_messages = get_session_history(
    session_id="test5",
    date=datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
)

runnable_with_history = RunnableWithMessageHistory(
    RunnablePassthrough.assign(messages_summarized=summarize_messages) | chain,
    get_session_history=lambda date, session_id: history_messages,
    input_messages_key="input",
    history_messages_key="history",
    history_factory_config=[
        ConfigurableFieldSpec(
            id="date",
            annotation=Date,
            name="Date",
            description=".",
            default="",
            is_shared=True,
        ),
        ConfigurableFieldSpec(
            id="session_id",
            annotation=str,
            name="Session ID",
            description="Unique identifier for the conversation.",
            default="",
            is_shared=True,
        ),
    ],
)




while True:
    print("You:")
    x = input()
    if x == "exit":
        break
    response = runnable_with_history.invoke(
        {
            "input": [HumanMessage(content=x)],
            "sumary": [
                AIMessage(content="你和用户随便聊了几句，用户告诉你他的名字是小牛")
            ],
        },
        config={
            "configurable": {"session_id": "test4"},
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
        },
    )

    print(f"AI:\n{response}")




# print(len(history_messages.messages))

# print(len(history_messages.get_messages()))
