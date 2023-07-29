import { Presets, SingleBar } from "cli-progress";
import {
    ensureDirSync,
    outputFileSync,
    readdirSync,
    readFileSync,
    removeSync,
} from "fs-extra";
import { lt, satisfies } from "semver";

import v0_4_11 from "@solc/0.4.11";
import v0_4_14 from "@solc/0.4.14";
import v0_4_15 from "@solc/0.4.15";
import v0_4_18 from "@solc/0.4.18";
import v0_4_19 from "@solc/0.4.19";
import v0_4_20 from "@solc/0.4.20";
import v0_4_23 from "@solc/0.4.23";
import v0_4_24 from "@solc/0.4.24";
import v0_4_25 from "@solc/0.4.25";
import v0_4_26 from "@solc/0.4.26";

import v0_5_0 from "@solc/0.5.0";
import v0_5_2 from "@solc/0.5.2";
import v0_5_3 from "@solc/0.5.3";
import v0_5_6 from "@solc/0.5.6";
import v0_5_7 from "@solc/0.5.7";
import v0_5_8 from "@solc/0.5.8";
import v0_5_9 from "@solc/0.5.9";
import v0_5_10 from "@solc/0.5.10";
import v0_5_11 from "@solc/0.5.11";
import v0_5_12 from "@solc/0.5.12";
import v0_5_13 from "@solc/0.5.13";
import v0_5_14 from "@solc/0.5.14";
import v0_5_15 from "@solc/0.5.15";
import v0_5_16 from "@solc/0.5.16";
import v0_5_17 from "@solc/0.5.17";

import v0_6_0 from "@solc/0.6.0";
import v0_6_1 from "@solc/0.6.1";
import v0_6_2 from "@solc/0.6.2";
import v0_6_4 from "@solc/0.6.4";
import v0_6_6 from "@solc/0.6.6";
import v0_6_7 from "@solc/0.6.7";
import v0_6_8 from "@solc/0.6.8";
import v0_6_9 from "@solc/0.6.9";
import v0_6_10 from "@solc/0.6.10";
import v0_6_11 from "@solc/0.6.11";
import v0_6_12 from "@solc/0.6.12";

import v0_7_0 from "@solc/0.7.0";
import v0_7_3 from "@solc/0.7.3";
import v0_7_4 from "@solc/0.7.4";
import v0_7_5 from "@solc/0.7.5";
import v0_7_6 from "@solc/0.7.6";

import v0_8_0 from "@solc/0.8.0";
import v0_8_1 from "@solc/0.8.1";
import v0_8_2 from "@solc/0.8.2";
import v0_8_3 from "@solc/0.8.3";
import v0_8_4 from "@solc/0.8.4";
import v0_8_5 from "@solc/0.8.5";
import v0_8_6 from "@solc/0.8.6";
import v0_8_7 from "@solc/0.8.7";

import { inputDir, outputDir, errorDir } from "./constant/constant.json";
import folderTree from "./constant/folder-tree.json";
import versions from "./constant/version-list.json";

type IVersion = {
    available: boolean;
    version: string;
    isNew?: boolean;
};

interface IOldInput {
    sources: {
        [key: string]: string;
    };
}

interface INewInput {
    language: "Solidity";
    sources: { [key: string]: { content: string } };
    settings: { outputSelection: { "*": { "*": string[] } } };
}

type IInput = IOldInput | INewInput;

const bar: SingleBar = new SingleBar({}, Presets.shades_classic);
const compilers = [
    v0_4_11,
    v0_4_14,
    v0_4_15,
    v0_4_18,
    v0_4_19,
    v0_4_20,
    v0_4_23,
    v0_4_24,
    v0_4_25,
    v0_4_26,
    v0_5_0,
    v0_5_2,
    v0_5_3,
    v0_5_6,
    v0_5_7,
    v0_5_8,
    v0_5_9,
    v0_5_10,
    v0_5_11,
    v0_5_12,
    v0_5_13,
    v0_5_14,
    v0_5_15,
    v0_5_16,
    v0_5_17,
    v0_6_0,
    v0_6_1,
    v0_6_2,
    v0_6_4,
    v0_6_6,
    v0_6_7,
    v0_6_8,
    v0_6_9,
    v0_6_10,
    v0_6_11,
    v0_6_12,
    v0_7_0,
    v0_7_3,
    v0_7_4,
    v0_7_5,
    v0_7_6,
    v0_8_0,
    v0_8_1,
    v0_8_2,
    v0_8_3,
    v0_8_4,
    v0_8_5,
    v0_8_6,
    v0_8_7,
];

function setup(): void {
    removeSync(outputDir);
    ensureDirSync(outputDir);
    removeSync(errorDir);
    ensureDirSync(errorDir);
}

function detect(source: string): IVersion {
    const pragmas = source.match(/pragma.solidity.*;/g);
    if (pragmas === null) throw new Error("No pragma solidity");

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

function buildInput(filename: string, source: string, isNew?: boolean): IInput {
    return isNew
        ? ({
              language: "Solidity",
              sources: { [filename]: { content: source } },
              settings: {
                  outputSelection: { "*": { "*": ["evm.bytecode"] } },
              },
          } as INewInput)
        : ({ sources: { [filename]: source } } as IOldInput);
}

function compile(
    filename: string,
    version: string,
    input: IInput,
    isNew?: boolean
): string {
    const index = versions.indexOf(version);
    if (index === -1) throw new Error(`Version ${version} is not avaiable`);

    const output = isNew
        ? JSON.parse(compilers[index].compile(JSON.stringify(input)))
        : compilers[index].compile(input, 1);

    if (output.errors) {
        let errorMessage = "";
        if (isNew) {
            for (const error of output.errors) {
                errorMessage += `\n${error.message}`;
            }
        } else {
            for (const error of output.errors) {
                errorMessage += `\n${error}`;
            }
        }
        return errorMessage;
    }

    const bytecode = isNew
        ? Object.keys(output.contracts[filename]).reduce(
              (pre, key) =>
                  pre + output.contracts[filename][key].evm.bytecode.object,
              ""
          )
        : Object.keys(output.contracts).reduce(
              (pre, key) => pre + output.contracts[key].bytecode,
              ""
          );

    const buffer = bytecode.match(/.{1,2}/g);
    if (!buffer) throw new Error("Bytecode null");

    return buffer.join(" ");
}

function main(): void {
    for (let vulnerability of folderTree) {
        console.log(`Start compile in ${vulnerability}`);
        const contracts = readdirSync(`${inputDir}/${vulnerability}`);
        bar.start(contracts.length, 0);

        for (let index = 0; index < contracts.length; index++) {
            bar.update(index + 1);
            const contract = contracts[index];
            const contractName = contract.replace(/\.[^/.]+$/, "");

            try {
                const contractSource = readFileSync(
                    `${inputDir}/${vulnerability}/${contract}`,
                    "utf8"
                );
                const { available, version, isNew } = detect(contractSource);

                if (!available)
                    throw new Error(`Version ${version} is not avaiable`);

                const input = buildInput(contractName, contractSource, isNew);
                const bytecode = compile(contractName, version, input, isNew);

                outputFileSync(
                    `${outputDir}/${vulnerability}/${contractName}.txt`,
                    bytecode
                );
            } catch (err) {
                outputFileSync(
                    `${errorDir}/${vulnerability}/${contractName}.txt`,
                    err.message
                );
            }
        }

        bar.stop();
    }
}

(() => {
    setup();
    main();
})();
