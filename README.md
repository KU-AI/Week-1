# KU-AI Github Guidlines

## Configuration

git config --global user.name "John Doe"

git config --global user.email johndoe@example.com

## Initialize Git

본인이 작성한 루트 디렉토리로 이동

git init

git remote add origin https://github.com/KU-AI/Week-1.git

git checkout -b lbs # 브랜치 생성과 전환을 함께할 때는 -b를 추가하고 아닐 경우에는 빼고, lbs(본인 이니셜)

## Git Commit Changes

git add .

git commit -m "변경 내용에 대한 설명"

git push origin lbs # 본인이 설정한 브랜치에 푸쉬