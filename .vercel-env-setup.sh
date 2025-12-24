#!/bin/bash
# Script to set Vercel environment variable

echo "Setting VITE_API_URL in Vercel..."
cd frontend
npx vercel env add VITE_API_URL production << ENVEOF
https://bryanhoulton--streaming-memory-familyassistant-serve.modal.run
ENVEOF

echo ""
echo "✅ Environment variable set!"
echo "Now redeploy with: cd frontend && npx vercel --prod"
