Project: Persona RAG Chatbot

Phase: Business Requirements Gathering

Author: Duc Lai

---

## 1. PROJECT OBJECTIVE

Develop a web-based chatbot system capable of providing accurate answers based on user-uploaded documents (PDF, DOCX, TXT).

The goal is to combine Information Retrieval (Retrieval) and Text Generation (Generation) to create a natural, trustworthy, and user-friendly question-answering experience for individual or small-group users.

---

## 2. PROJECT SCOPE

- **User scope:** Designed for individual users or small groups of 3–5 concurrent users.
- **Core features:** Document upload, intelligent chat, conversation history, system settings.
- **Technical scope:** Implemented as a web application, deployable locally or via Docker containers.
- **Data scope:** Handles private user documents, not publicly shared.

---

## 3. BUSINESS REQUIREMENTS

| **Category** | **Description** |
| --- | --- |
| **Documents** | Users can upload, store, and delete their documents. The system manages and isolates user-specific data. |
| **Conversation** | Users can input questions and receive chatbot responses derived from uploaded data, with document citations. The chatbot understands conversation context and stores chat history for later access. |
| **Chat** | Users can create, rename, or delete chat sessions. |
| **Settings** | Users can customize RAG system parameters such as embedding models, LLM providers, and generation models. |
| **Performance & Security** | The system ensures stability, quick response time, secure data handling, and rollback mechanisms upon failure. Admin can review activity logs. |
| **Vector data maintenance** | Only new data is processed upon upload (text extraction → chunking → embedding → vector storage). Deletion removes only related vectors without affecting others. |

---

## 4. FUNCTIONAL REQUIREMENTS

| **Functional Group** | **Detailed Description** |
| --- | --- |
| **1. Document Management** | - Upload, delete, and store user documents.- Securely store original files in a protected database.- Process only new data when uploading. |
| **2. Intelligent Chatbot Querying** | - Accept user questions.- Retrieve and synthesize information from vector store.- Generate answers with proper document citations. |
| **3. Chat Management** | - Create, rename, and delete chat sessions.- Store both current and previous conversation histories. |
| **4. System Configuration (Settings)** | - Allow model selection for embeddings and LLMs.- Save the latest configuration preferences per user. |
| **5. Administration** | - Enable admin to view system logs and error traces. |

---

## 5. NON-FUNCTIONAL REQUIREMENTS

| **Category** | **Description** |
| --- | --- |
| **Performance** | - Chatbot response time < 5 seconds.- Fast and stable document upload + processing.- Support for 3–5 concurrent users. |
| **Security** | - Uploaded documents encrypted and isolated per user.- Secure communication between frontend and backend (HTTPS or token-based).- No document sharing between users. |
| **Stability & Fault Recovery** | - System should not crash; rollback mechanism when errors occur.- Ensure frontend-backend consistency. |
| **Deployment & Operations** | - Deployable both locally and via Docker containers.- Maintain logs for monitoring and debugging. |
| **User Experience** | - Smooth, responsive UI with clear feedback and fast interactions. |

---