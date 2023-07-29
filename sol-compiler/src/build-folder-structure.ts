import {
    createWriteStream,
    readdirSync,
    remove,
    removeSync,
    statSync,
} from "fs-extra";

import { inputDir } from "./constant/constant.json";

function main(): string[] {
    const stack = readdirSync(inputDir);
    const output: string[] = [];

    while (stack.length) {
        const currentNode = stack.pop();
        if (!currentNode) break;

        output.push(currentNode);

        const children = readdirSync(`${inputDir}/${currentNode}`);
        let flag: boolean = false;
        for (let child of children) {
            const childPath = `${currentNode}/${child}`;
            if (statSync(`${inputDir}/${childPath}`).isDirectory()) {
                stack.push(childPath);
                flag = true;
            }
        }

        if (flag) output.pop();
    }

    return output;
}

(() => {
    const tree = main();

    removeSync("src/constant/folder-tree.json");
    const writeStream = createWriteStream("src/constant/folder-tree.json");

    writeStream.write("[");
    for (let i = 0; i < tree.length - 1; i++) {
        writeStream.write(`\n\t"${tree[i]}",`);
    }
    writeStream.write(`\n\t"${tree.pop()}"`);
    writeStream.write("\n]");
})();
