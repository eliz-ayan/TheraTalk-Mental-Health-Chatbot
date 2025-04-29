
source ~/rasa_env_311/bin/activate

osascript <<EOF
tell application "Terminal"
    activate
    do script "cd $(pwd); source ~/rasa_env_311/bin/activate; rasa run actions"
end tell
EOF

=
osascript <<EOF
tell application "Terminal"
    activate
    do script "cd $(pwd); source ~/rasa_env_311/bin/activate; rasa run --enable-api --cors '*' --debug"
end tell
EOF

=
open index.html

echo "🚀 TheraTalk is now running. Your action server, Rasa server, and frontend are launching..."
