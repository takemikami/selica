#!/bin/bash

if [ "$TRAVIS_REPO_SLUG" == "takemikami/selica" ] && [ "$TRAVIS_PULL_REQUEST" == "false" ] && [ "$TRAVIS_BRANCH" == "master" ]; then

  echo -e "Publishing scaladoc...\n"

  cp -R "target/scala-2.11/api" $HOME/scaladoc-latest

  cd $HOME
  git config --global user.email "travis@travis-ci.org"
  git config --global user.name "travis-ci"
  git clone --quiet --branch=gh-pages https://${GH_TOKEN}@github.com/takemikami/selica gh-pages > /dev/null

  cd gh-pages
  git rm -rf ./scaladoc
  cp -Rf $HOME/scaladoc-latest ./scaladoc
  git add -f .
  git commit -m "Latest scaladoc on successful travis build $TRAVIS_BUILD_NUMBER auto-pushed to gh-pages"
  git push -fq origin gh-pages > /dev/null

  echo -e "Published scaladoc to gh-pages.\n"

fi
