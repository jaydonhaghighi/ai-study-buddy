## Prerequisites

- Node.js (v16 or higher)
- npm or yarn
- A Firebase project (create one at [Firebase Console](https://console.firebase.google.com/))

## Setup Instructions

### Install Dependencies

```bash
npm install
```

### Configure Firebase

1. **Create a Firebase Project** (if you haven't already):
   - Go to [Firebase Console](https://console.firebase.google.com/)
   - Create a new project or select an existing one

2. **Set up Environment Variables**:
   - Copy `.env.example` to `.env`:
     ```bash
     cp .env.example .env
     ```
   - Update `.env` with your Firebase configuration from Firebase Console > Project Settings
   - **Add OpenAI API Key**:
     - Get your API key from [OpenAI Platform](https://platform.openai.com/api-keys)
     - Add it to your `.env` file:
       ```env
       VITE_OPENAI_API_KEY=your_openai_api_key_here
       ```
     - **Important**: For production, use a backend server to keep your API key secure. Exposing it in the client is not recommended for production use.

3. **Enable Firebase Services**:
   - **Authentication**: Go to Authentication > Sign-in method and enable Email/Password
   - **Firestore Database**: Go to Firestore Database and create a database
   - **Storage**: Go to Storage and get started

4. **Configure Firestore Security Rules**:
   - Go to [Firestore Rules](https://console.firebase.google.com/project/YOUR_PROJECT_ID/firestore/rules)
   - Replace the default rules with:
     ```javascript
     rules_version = '2';
     service cloud.firestore {
       match /databases/{database}/documents {
         match /{document=**} {
           allow read, write: if request.auth != null;
         }
       }
     }
     ```
   - Click **"Publish"** to save the rules
   - **Note**: These rules allow any authenticated user to read/write. For production, add more specific rules.

5. **Configure Storage Security Rules** (Optional):
   - Go to Storage > Rules
   - Update rules to allow authenticated users:
     ```javascript
     rules_version = '2';
     service firebase.storage {
       match /b/{bucket}/o {
         match /uploads/{userId}/{allPaths=**} {
           allow read, write: if request.auth != null && request.auth.uid == userId;
         }
       }
     }
     ```

## Running the Application

### Development Mode

```bash
npm run dev
```

This will start a development server (usually at `http://localhost:5173`).

### Build for Production

```bash
npm run build
```

