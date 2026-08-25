import React from 'react';
import ReactDOM from 'react-dom/client';
import App from '@/app/App';
import {
  EDGE_DATABASE_FORMAT_IDENTITY,
  EDGE_DATABASE_SCHEMA_VERSION,
} from '@/shared/releaseIdentity';
import './styles.css';

document.documentElement.dataset.seeonEdgeFormat = EDGE_DATABASE_FORMAT_IDENTITY;
document.documentElement.dataset.seeonEdgeSchema = String(EDGE_DATABASE_SCHEMA_VERSION);

ReactDOM.createRoot(document.getElementById('root') as HTMLElement).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
