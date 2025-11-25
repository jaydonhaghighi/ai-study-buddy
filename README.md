## Prerequisites

- Node.js (v20 or higher)
- npm or yarn
- A Firebase project (create one at [Firebase Console](https://console.firebase.google.com/))
- Firebase Blaze Plan (Pay-as-you-go) - Required for Cloud Functions and Vertex AI

## Setup Instructions

### Install Dependencies

```bash
# Install frontend dependencies
npm install

# Install Cloud Functions dependencies
cd functions
npm install
cd ..
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
   - **Add Google AI API Key** (for Genkit/Vertex AI):
     - Go to [Google AI Studio](https://makersuite.google.com/app/apikey) or enable Vertex AI API in [Google Cloud Console](https://console.cloud.google.com/)
     - Get your API key
     - Add it to your `.env` file:
       ```env
       GOOGLE_GENAI_API_KEY=your_google_ai_api_key_here
       ```
     - **For Cloud Functions**: Set this as a secret in Firebase:
       ```bash
       firebase functions:secrets:set GOOGLE_GENAI_API_KEY
       ```

3. **Enable Firebase Services**:
   - **Authentication**: Go to Authentication > Sign-in method and enable Email/Password
   - **Firestore Database**: Go to Firestore Database and create a database
   - **Cloud Functions**: Go to Functions and enable (requires Blaze plan)
   - **Vertex AI**: Go to Build > Vertex AI in Firebase and enable (requires Blaze plan)

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
# Start frontend dev server
npm run dev

# In another terminal, start Cloud Functions emulator (optional)
cd functions
npm run serve
```

The frontend will run at `http://localhost:5173`.

### Deploy Cloud Functions

```bash
# Build functions
cd functions
npm run build

# Deploy to Firebase
firebase deploy --only functions
```

### Build for Production

```bash
# Build frontend
npm run build

# Deploy frontend and functions
firebase deploy
```

## Architecture

This application uses:
- **Frontend**: React + TypeScript + Vite
- **Backend**: Firebase Cloud Functions with Genkit
- **AI**: Google Gemini 1.5 Flash (via Vertex AI)
- **Database**: Cloud Firestore (for messages and sessions)
- **Authentication**: Firebase Auth

The chat uses **Genkit's persistent chat sessions**, which automatically manage conversation history and context across multiple chat sessions per user.

