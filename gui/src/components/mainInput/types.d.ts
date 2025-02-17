import { ContextProviderDescription } from "core";

export type ComboBoxItemType =
  | "contextProvider"
  | "slashCommand"
  | "file"
  | "query";

export interface ComboBoxItem {
  title: string;
  description: string;
  id?: string;
  content?: string;
  type: ComboBoxItemType;
  contextProvider?: ContextProviderDescription;
  query?: string;
  label?: string;
}
