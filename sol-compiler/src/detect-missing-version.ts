import { Presets, SingleBar } from "cli-progress";
import { readdirSync, readFileSync } from "fs-extra";
import { lt, satisfies } from "semver";

import { inputDir } from "./constant/constant.json";
import folderTree from "./constant/folder-tree.json";
import versions from "./constant/version-list.json";

type IVersion = {
    available: boolean;
    version: string;
    isNew?: boolean;
};

const bar: SingleBar = new SingleBar({}, Presets.shades_classic);

function detect(source: string): IVersion {
    const pragmas = source.match(/pragma.solidity.*;/g);
    if (pragmas === null) throw "No pragma solidity";

    let range = "";
    const pragmaSet: Set<string> = new Set(pragmas);
    pragmaSet.forEach(
        (value) =>
            (range =
                range +
                value.slice(16, value.indexOf("//")).replace(";", "").trim() +
                " ")
    );

    for (let version of versions) {
        if (satisfies(version, range)) {
            return { available: true, version, isNew: !lt(version, "0.5.0") };
        }
    }

    return { available: true, version: range };
}

function main(): Set<string> {
    const missingVersion: Set<string> = new Set();

    for (const path of folderTree) {
        console.log(`Start detect in ${path}`);

        const contracts = readdirSync(`${inputDir}/${path}`);
        bar.start(contracts.length, 0);

        for (let i = 0; i < contracts.length; i++) {
            bar.update(i + 1);
            try {
                const contractSource = readFileSync(
                    `${inputDir}/${path}/${contracts[i]}`,
                    "utf8"
                );
                const { available, version } = detect(contractSource);
                if (!available) missingVersion.add(version);
            } catch (err) {}
        }

        bar.stop();
    }

    return missingVersion;
}

(() => {
    const missingVersion = main();
    console.log(missingVersion);
})();
