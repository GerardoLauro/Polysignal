# Third-party frontend libraries

The dashboard bundles these libraries so that it can run without a CDN after installation. Their original JavaScript files and embedded notices are preserved.

| Library | Bundled version | License | Source |
| --- | --- | --- | --- |
| React | 18.3.1 | [MIT](webapp/vendor/react-LICENSE.txt) | [React release](https://github.com/facebook/react/tree/v18.3.1) |
| ReactDOM | 18.3.1 | [MIT](webapp/vendor/react-LICENSE.txt) | [React release](https://github.com/facebook/react/tree/v18.3.1) |
| Babel standalone | 7.29.0 | [MIT](webapp/vendor/babel-LICENSE.txt) | [Babel release](https://github.com/babel/babel/tree/v7.29.0) |

ReactDOM also retains its embedded Modernizr and regenerator notices. Python dependencies retain their own licenses; they are downloaded during installation and are not vendored here.

The project [MIT license](LICENSE) covers the project code as originally supplied. It does not automatically grant redistribution rights over third-party social text, assignment materials, or referenced publications. Dataset licenses and platform terms must be considered separately by anyone distributing data.
