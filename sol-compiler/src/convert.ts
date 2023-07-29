import { resolve } from "path";
import {
    ensureDirSync,
    readdirSync,
    readFileSync,
    removeSync,
    createWriteStream,
    WriteStream,
} from "fs-extra";
import { stringify, Stringifier } from "csv-stringify";

import folderTree from "../folder-tree.json";
import { outputDir, convertDir } from "../constant.json";
import { Presets, SingleBar } from "cli-progress";

const bar: SingleBar = new SingleBar({}, Presets.shades_classic);
const columns = ["ADDRESS", "BYTECODE", "LABEL"];

function setup(): void {
    removeSync(convertDir);
    ensureDirSync(convertDir);
}

let vulnerability: string,
    vulnerabilityName: string;
let contracts: string[],
    contract: string,
    contractName: string,
    contractPath: string,
    contractSource: string;
let index: number;
let writeableStream: WriteStream;
let stringifier: Stringifier;

function main(): void {
    for (vulnerability of folderTree) {
        console.log(`Start conver in ${vulnerability}`);
        vulnerabilityName = vulnerability.split("/").pop();

        writeableStream = createWriteStream(resolve(convertDir, vulnerabilityName + ".csv"));
        stringifier = stringify({ header: true, columns: columns });

        contracts = readdirSync(resolve(outputDir, vulnerability));
        bar.start(contracts.length, 0);

        for (index = 0; index < contracts.length; index++) {
            bar.update(index + 1);

            try {
                contract = contracts[index];
                contractName = contract.replace(/\.[^/.]+$/, "");
                contractPath = resolve(outputDir, vulnerability, contract);
                contractSource = readFileSync(contractPath, "utf8");

                stringifier.write([contractName, contractSource, "1"]);
            } catch (err) {
            } finally {
                contract = null;
                contractName = null;
                contractPath = null;
                contractSource = null;
            }
        }

        bar.stop();
        stringifier.pipe(writeableStream);
    }
}

setup();
main();
