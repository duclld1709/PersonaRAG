"use client";

import clsx from "clsx";
import { ChevronLeft, ChevronRight } from "lucide-react";
import { ButtonHTMLAttributes, forwardRef } from "react";

type SidebarToggleProps = Omit<
  ButtonHTMLAttributes<HTMLButtonElement>,
  "type" | "aria-expanded"
> & {
  isOpen: boolean;
  controlledId: string;
};

export const SidebarToggle = forwardRef<HTMLButtonElement, SidebarToggleProps>(
  ({ isOpen, controlledId, className, ...props }, ref) => {
    const Icon = isOpen ? ChevronLeft : ChevronRight;

    return (
      <button
        ref={ref}
        type="button"
        aria-expanded={isOpen}
        aria-controls={controlledId}
        className={clsx(
          "flex h-10 w-10 items-center justify-center rounded-full border border-border bg-card text-text shadow-sm transition motion-safe:duration-200 motion-safe:ease-in-out hover:border-primary hover:bg-muted focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary",
          className
        )}
        {...props}
      >
        <Icon className="h-5 w-5" strokeWidth={1.75} aria-hidden="true" />
        <span className="sr-only">{isOpen ? "Collapse sidebar" : "Expand sidebar"}</span>
      </button>
    );
  }
);

SidebarToggle.displayName = "SidebarToggle";

export default SidebarToggle;
