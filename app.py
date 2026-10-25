import uuid
import streamlit as st
from openai import OpenAI, AuthenticationError, PermissionDeniedError, RateLimitError, APIConnectionError, APIStatusError
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

st.set_page_config(page_title='Mini GPT', page_icon='🤖', layout='wide')
st.title('🤖 Mini GPT')
st.caption('Chat with OpenAI using your own API key')

if 'chats' not in st.session_state:
    st.session_state.chats = {}
if 'current_chat' not in st.session_state:
    st.session_state.current_chat = None
if 'verified_key' not in st.session_state:
    st.session_state.verified_key = None
if 'api_key_input' not in st.session_state:
    st.session_state.api_key_input = ''

def new_chat():
    chat_id = str(uuid.uuid4())
    st.session_state.chats[chat_id] = {'title': 'New Chat', 'messages': []}
    st.session_state.current_chat = chat_id

def verify_key(key):
    # A real authenticated request. Does not incur model-generation tokens.
    OpenAI(api_key=key, timeout=15.0, max_retries=0).models.list()

with st.sidebar:
    st.header('🔑 OpenAI API access')
    # Streamlit's password field masks the key. Chrome may still offer to save it;
    # that popup is controlled by the browser, not Streamlit.
    entered_key = st.text_input('OpenAI API key', type='password', key='api_key_input',
                                placeholder='Paste your API key')
    key = entered_key.strip()
    if key != st.session_state.verified_key:
        st.session_state.verified_key = None

    if st.button('Verify API Key', type='primary', use_container_width=True):
        if not key:
            st.error('Enter an OpenAI API key first.')
        else:
            with st.spinner('Checking your API key...'):
                try:
                    verify_key(key)
                except AuthenticationError:
                    st.error('Invalid API key. Enter a valid OpenAI API key.')
                except PermissionDeniedError:
                    st.error('This API key does not have the required permissions.')
                except RateLimitError:
                    st.error('OpenAI rate limit or quota issue. Check your API account.')
                except (APIConnectionError, APIStatusError) as exc:
                    st.error(f'Could not verify the key right now: {exc}')
                except Exception as exc:
                    st.error(f'Verification failed: {exc}')
                else:
                    st.session_state.verified_key = key
                    st.success('API key verified. Chat is unlocked!')

    verified = bool(key) and key == st.session_state.verified_key
    if verified:
        st.success('✓ API key verified')
    else:
        st.info('Verify your API key to unlock the question box.')

    model = st.selectbox('Model', ['gpt-4.1-mini', 'gpt-4.1'])
    system_prompt = st.text_area('System role', value='You are a helpful, accurate AI assistant.')
    st.divider()
    st.subheader('💬 Chat history')
    if st.button('➕ New Chat', use_container_width=True):
        new_chat()
        st.rerun()
    for chat_id, chat_data in list(st.session_state.chats.items()):
        if st.button(chat_data['title'], key=f'chat_{chat_id}', use_container_width=True):
            st.session_state.current_chat = chat_id
            st.rerun()
    if st.button('🗑️ Clear All Chats', use_container_width=True):
        st.session_state.chats = {}
        st.session_state.current_chat = None
        st.rerun()

if st.session_state.current_chat is None:
    new_chat()
chat = st.session_state.chats[st.session_state.current_chat]

for message in chat['messages']:
    with st.chat_message('user' if isinstance(message, HumanMessage) else 'assistant'):
        st.markdown(message.content)

if not verified:
    st.warning('🔒 Enter and verify a valid API key in the sidebar before asking questions.')

question = st.chat_input('Ask Mini GPT anything...', disabled=not verified)
if question and verified:
    with st.chat_message('user'):
        st.markdown(question)
    pending_messages = chat['messages'] + [HumanMessage(content=question)]
    with st.chat_message('assistant'):
        with st.spinner('Thinking...'):
            try:
                llm = ChatOpenAI(model=model, api_key=key, timeout=45, max_retries=1)
                response = llm.invoke([SystemMessage(content=system_prompt)] + pending_messages)
                answer = response.content
                if not isinstance(answer, str):
                    answer = str(answer)
                if not answer:
                    raise ValueError('The model returned an empty response.')
            except AuthenticationError:
                st.session_state.verified_key = None
                st.error('Authentication failed. Please verify your API key again.')
            except PermissionDeniedError:
                st.error('This key cannot use the selected model. Check model permissions.')
            except RateLimitError:
                st.error('Rate limit or insufficient API quota. Check OpenAI billing.')
            except Exception as exc:
                st.error(f'Unable to generate an answer: {exc}')
            else:
                st.markdown(answer)
                chat['messages'] = pending_messages + [AIMessage(content=answer)]
                if chat['title'] == 'New Chat':
                    chat['title'] = question[:35]
