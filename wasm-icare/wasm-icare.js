/**
 * @module wasm-icare
 */

/**
 * URL to the Pyodide CDN
 * @type {string}
 */
let pyodideCDNURL = 'https://cdn.jsdelivr.net/pyodide/v0.23.2/full/pyodide.js';

/**
 * Global variable to hold the instance of Pyodide
 * @type {object}
 */
let pyodide = null;

/**
 * Function to load Pyodide from the CDN.
 * @async
 * @function
 * @returns {Promise<void>}
 */
async function loadPyodideFromCDN() {
    if (!pyodide) {
        const script = document.createElement('script');
        script.src = pyodideCDNURL;
        document.body.appendChild(script);

        await new Promise((resolve) => {
            script.onload = resolve;
        });

        pyodide = await window.loadPyodide();
    }
}

/**
 * Function to load files from a list of URLs and write them to the Pyodide file system.
 * @param fileURLs
 * @returns {Promise<Awaited<unknown>[]>}
 */
async function fetchFilesAndWriteToPyodideFS(fileURLs) {
    if (!pyodide) {
        throw new Error('Pyodide is not loaded. Please initialize this library using the loadWasmICare() function.');
    }

    async function fetchAndWriteFile(url) {
        try {
            const response = await fetch(url);

            if (!response.ok) {
                console.error(`Failed to fetch file from ${url}`);
                return {isError: true, message: `Failed to fetch file from ${url}`};
            }

            const fileContent = await response.text();
            const fileName = url.substring(url.lastIndexOf('/') + 1);
            pyodide.FS.writeFile(fileName, fileContent);

            console.log(`File ${fileName} successfully loaded to the Pyodide file system.`);
            return {isError: false, message: `File ${fileName} successfully loaded to the Pyodide file system.`};
        } catch (error) {
            console.error(`Error fetching and writing file: ${error.message}`);
            return {isError: true, message: `Error fetching and writing file: ${error.message}`};
        }
    }

    return await Promise.all(fileURLs.map(fetchAndWriteFile));
}

/**
 * Function to load the iCARE Python package and convert it into Wasm. Return the Wasm-iCARE object.
 * @returns {Promise<*>}
 */
async function loadICare() {
    if (!pyodide) {
        throw new Error('Pyodide is not loaded. Please initialize this library using the initialize() function.');
    }

    await pyodide.loadPackage('micropip');
    const micropip = pyodide.pyimport('micropip');
    await micropip.install('pyicare');

    return pyodide.runPython(`import icare
icare`);
}

/**
 * Function to initialize Wasm-iCARE.
 * @async
 * @function
 * @returns {Promise<void>}
 */
async function loadWasmICare() {
    await loadPyodideFromCDN();
    const icare = await loadICare();
    return icare;
}

export {
    loadWasmICare,
    pyodide
};