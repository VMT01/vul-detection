## This tool will detect and compile every solidity file with provided versions.

<hr>

## Requirements
Use this command to install dependencies
```bash
yarn
```
> Remember to edit `constant.json` file before run

<br>

## Before start
Run `yarn build-folder-tree` to create folder tree for detect version and compile, etc. Then remove parent folder path in `folder-tree.txt` to avoid error 

<i>(currently still cannot find any solution for this)</i>

<br>

## Detect version first
Run `yarn detect-version` to find all solidity version.Then add them to `/src/constant.json` by your hand :)

<br>

## It's time to compile
Run `yarn compile` for the magic. And pray for it. Pls dont error