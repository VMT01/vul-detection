import { execSync } from "child_process";
import { createWriteStream, ensureDir, removeSync } from "fs-extra";

import versions from "./constant/version-list.json";

(() => {
    removeSync("src/@types/solc.d.ts");
    ensureDir("src/@types");
    const writeStream = createWriteStream("src/@types/solc.d.ts");

    let command: string = `yarn add @solc/${versions[0]}@npm:solc@${versions[0]}`;
    execSync(command);
    writeStream.write(`declare module '@solc/${versions[0]}';`);

    for (let i = 1; i < versions.length; i++) {
        command = `yarn add @solc/${versions[i]}@npm:solc@${versions[i]}`;
        execSync(command);
        writeStream.write(`\ndeclare module '@solc/${versions[i]}';`);
    }
})();
